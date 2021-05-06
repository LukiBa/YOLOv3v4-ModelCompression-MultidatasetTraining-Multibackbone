# %%
from __future__ import division

from models import *
from utils import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import pathlib
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3tiny/yolov3-tiny.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='./data/coco2014.data', help='*.data path')    
    parser.add_argument('--weights', type=str, default='./weights/yolov3-tiny.weights', help='path to weights file')
    parser.add_argument('--model_out',  type=str, default='./pt_models/yolov3-tiny-int8.pt', help='path to output')
    parser.add_argument('--scale_out',  type=str, default='./pt_models/yolov3-tiny-int8_scales.txt', help='path to output')
    parser.add_argument('--q_bits', type=float, default=8, help='number of used bits')
    parser.add_argument('--img_size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    return parser.parse_args()

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma


    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv




# %%
# # Stats Functions
class Quant_Method:
    Min_Max = 1
    Bias_Correction = 2
    Power_Two = 3
    Mean = 4
    Two_Third = 5
    Remove_Outlier = 6

# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key, inout):
    max_val = torch.max(torch.abs(x))
  
    if key not in stats:
        stats[key] = {"input": {"max": 0, "total": 0},"output": {"max": 0, "total": 0}}
        stats[key][inout]['max'] = max_val
        stats[key][inout]['total'] += 1
    else:
        if stats[key][inout]['max'] < max_val:
            stats[key][inout]['max'] = max_val
        stats[key][inout]['total'] += 1
        

    return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats, targets=None):

  img_dim = x.shape[2]
  loss = 0
  layer_outputs, yolo_outputs = [], []

  for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):

    

    if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
        stats = updateStats(x.clone(), stats, str(module_def["type"]) + "_" + str(i),"input")
        if module_def["type"] in ["convolutional"]:
            if int(module_def["batch_normalize"]) > 0:
                x = module[0](x)                    
                x = module[2](x)
            else:
                x = module(x)
        else:
            x = module(x)
        stats = updateStats(x.clone(), stats, str(module_def["type"]) + "_" + str(i),"output")

    elif module_def["type"] == "route":
        x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
    elif module_def["type"] == "shortcut":
        layer_i = int(module_def["from"])
        x = layer_outputs[-1] + layer_outputs[layer_i]
    elif module_def["type"] == "yolo":
        x, layer_loss = module[0](x, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
    layer_outputs.append(x)

  return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader, num_bits):

    model.eval()
    stats = {}
    i = 0

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Gathering stats")):
      # Extract labels

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            i += 1
            if i < 25:
                stats = gatherActivationStats(model, imgs, stats)

    final_stats = {}

    qmax = ((2**num_bits)/2)-1
    qmin = -(qmax+1)

    for key, value in stats.items():
        final_stats[key] = {"s_x" : qmax/(value["input"]["max"]),"s_y" : qmax/(value["output"]["max"])}

    return final_stats


# %%
from quant_util import *
def quantize_layer(layer, num_bits, S, name, quant_method):
    
    if quant_method == Quant_Method.Min_Max:
        max_w = torch.max(torch.abs(layer.weight.data))
    if quant_method == Quant_Method.Bias_Correction:
        print("Not implemented yet")
    if quant_method == Quant_Method.Power_Two:
        scale = power_two(layer.weight.data.cpu(),S[name],num_bits)
    if quant_method == Quant_Method.Mean:
        layer.weight.data = mean_scale(layer.weight.data)
        max_w = torch.max(torch.abs(layer.weight.data))
    if quant_method == Quant_Method.Two_Third:
        layer.weight.data = squeeze_net(layer.weight.data)
        max_w = torch.max(torch.abs(layer.weight.data))
    if quant_method == Quant_Method.Remove_Outlier:
        layer.weight.data = reject_outliers(layer.weight.data,3)
        max_w = torch.max(torch.abs(layer.weight.data))


    qmax = ((2**num_bits)/2)-1
    qmin = -(qmax+1)

    if layer.bias is not None:
        max_b = torch.max(torch.abs(layer.bias.data))
        S[name]["s_b"] = qmax/max_b

   

    if quant_method != Quant_Method.Power_Two:
        S[name]["s_w"] = qmax/max_w
    else:
        S[name]["s_w"] = scale
    
    
    layer.weight.data = (torch.mul(layer.weight.data,S[name]["s_w"]))

    layer.weight.data.clamp_(qmin, qmax).round_()

    if layer.bias is not None:
        layer.bias.data = (torch.mul(layer.bias.data,(S[name]["s_w"]*S[name]["s_x"])))
    
    
    if layer.bias is not None:
        if(torch.max(torch.abs(layer.bias.data)) > qmax):
            k = torch.max(torch.abs(layer.bias.data))
            print("bias" + str(k))
        layer.bias.data.round_()#.clamp_(qmin, qmax).round_()

    return layer


# %%
# # Quantization Functions
from quant_util import *
def quantize_tensor(x, num_bits, scale):
    
    qmax = (2.**num_bits - 1.)/2.
    qmax = math.floor(qmax)
    qmin = -(qmax+1)

    q_x = x
    q_x.mul_(scale)
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x

def dequantize_tensor(q_x, scale):
    return (q_x.float() / scale)

def calc_adjust_layer(x, layer, S_layer,num_bits, shift_scale):
    
    qmax = (2.**num_bits - 1.)/2.
    qmax = math.floor(qmax)
    qmin = -(qmax+1)
    a = layer(x)
    #a = F.relu(a)

    #scale output from layer
    adjustment = (S_layer["s_y"]/(S_layer["s_w"]*S_layer["s_x"]))
    #print(str(S_layer["s_x"]) + " " + str(S_layer["s_y"]) + " " + str(S_layer["s_w"]))
    #print(adjustment)
    if shift_scale:
        #GEMMLOWPE
        shift,scale = findShiftScale(adjustment.cpu(),num_bits)
        a = scale * a # Integer will be multiplied; Shift out the unnecessary bits
        a  = ((2.**shift) * a) 
    else:
        #print(adjustment)
        #WITHOUT GEMMLOPE

        a = a * adjustment 

    a.clamp_(qmin, qmax).round_()
    return a

# %%
# # Forwarding
def quantForward(model, x, stats, num_bits, shift_scale):
    img_dim = x.shape[2]
    loss = 0
    layer_outputs, yolo_outputs = [], []

    x = quantize_tensor(x, num_bits,S["convolutional_0"]["s_x"])

    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
            if module_def["type"] in ["convolutional"]:
                if int(module_def["batch_normalize"]) > 0:
                    x = calc_adjust_layer(x, module[0], stats[str("convolutional") + "_" + str(i)],num_bits,shift_scale)
                    x = module[2](x)
                else:
                    if i == 15 or i == 22:
                        x = module[0](x)
                        x = dequantize_tensor(x, (S[str("convolutional") + "_" + str(i)]["s_w"]*S[str("convolutional") + "_" + str(i)]["s_x"]))                       
                  
                    else:
                        x = calc_adjust_layer(x, module[0], stats[str("convolutional") + "_" + str(i)],num_bits,shift_scale)

            else:
                x = module(x)
        elif module_def["type"] == "route":
            
            x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)

        elif module_def["type"] == "shortcut":
            layer_i = int(module_def["from"])
            x = layer_outputs[-1] + layer_outputs[layer_i]
        elif module_def["type"] == "yolo":
            x, layer_loss = module[0](x, None, img_dim)
            loss += layer_loss
            yolo_outputs.append(x)
        layer_outputs.append(x)
    yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
    return yolo_outputs# if targets is None else (loss, yolo_outputs)


def forward(model, x):
    targets = None
    img_dim = x.shape[2]
    loss = 0
    layer_outputs, yolo_outputs = [], []
    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
            if module_def["type"] in ["convolutional"] and int(module_def["batch_normalize"]) > 0:
                x = module[0](x)
                x = module[2](x)
            else:
                x = module(x)
        elif module_def["type"] == "route":
            x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
        elif module_def["type"] == "shortcut":
            layer_i = int(module_def["from"])
            x = layer_outputs[-1] + layer_outputs[layer_i]
        elif module_def["type"] == "yolo":
            x, layer_loss = module[0](x, targets, img_dim)
            loss += layer_loss
            yolo_outputs.append(x)
        layer_outputs.append(x)
    yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
    return yolo_outputs if targets is None else (loss, yolo_outputs)


def testQuant(model, dataloader, num_bits, quant=False, stats=None, shift_scale=False):
    model.eval()

    data_config = parse_data_config("config/coco.data")
    valid_path = data_config["valid"]

    path=valid_path
    iou_thres=0.5
    conf_thres=0.001
    nms_thres=0.5
    img_size=416

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            if quant:
                outputs = quantForward(model, imgs, stats, num_bits = num_bits, shift_scale=shift_scale)
            else:
                outputs = forward(model,imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")


# %% [markdown]
if __name__ == "__main__":
    opt = _create_parser()
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model

    data_config = parse_data_config(opt.data)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    cfg = pathlib.Path(opt.cfg).absolute()
    model = Darknet(cfg).to(device)
    model.load_darknet_weights(opt.weights)

    import copy
    a_model = copy.deepcopy(model)

    # print("All modules")
    # for i, (module_def, module) in enumerate(zip(a_model.module_defs, a_model.module_list)):
    #     print(module)

    for i, (module_def, module) in enumerate(zip(a_model.module_defs, a_model.module_list)):
        if module_def["type"] in ["convolutional"]:
            if(int(module_def["batch_normalize"]) > 0):
                a_model.module_list[i][0] = fuse(module[0], module[1])
                #print(module)

    batch_size = opt.batch_size

    data_path = pathlib.Path(opt.data).absolute()
    data_config = parse_data_config(data_path)
    path = data_config["valid"]
    class_names = load_classes(data_config["names"])


    num_bits = opt.q_bits
    dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn
    )
  

    #testQuant(a_model, dataloader, num_bits = 8, quant=False) #testing without quantization,

    import copy
    q_model = copy.deepcopy(a_model)

    S = gatherStats(q_model, dataloader, num_bits)
    
    scale_path = pathlib.Path(opt.scale_out).absolute()
    f = open(scale_path, "w")
    #Quantize Layer
    for i, (module_def, module) in enumerate(zip(q_model.module_defs, q_model.module_list)):
        if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
            if module_def["type"] in ["convolutional"]:
                q_model.module_list[i][0] = quantize_layer(module[0],num_bits, S, module_def["type"] + "_" + str(i), Quant_Method.Power_Two)
                S_layer = S[module_def["type"] + "_" + str(i)]
                f.write(module_def["type"] + "_" + str(i) + "_adjustment;" + str(math.pow(2, math.ceil(math.log(S_layer["s_y"]/(S_layer["s_w"]*S_layer["s_x"]))/math.log(2))))+ "\r\n")
    f.close()
    print(S)
    torch.save(q_model.state_dict(),opt.model_out)

    #testQuant(q_model, dataloader,num_bits = num_bits, quant=True,stats=S, shift_scale=True)

#%%
# 



# print(S)

# testQuant(q_model, test_loader,num_bits = num_bits, quant=True,stats=S, shift_scale=True)

