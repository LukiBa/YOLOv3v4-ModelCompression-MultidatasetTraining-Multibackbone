# Author:Lukas Baischer
import argparse
import pathlib
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import numpy as np
import torch

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3tiny/yolov3-tiny-quant.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/rt.pt', help='path to weights file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--quantized', type=int, default=6,
                        help='0:quantization way one Ternarized weight and 8bit activation')
    parser.add_argument('--param_outpath', type=str, default='./parameters/int8_6')#'./detect_imgs', help='Path to output images. If set to None images are not saved.') #'./detect_imgs'
    parser.add_argument('--out_weights', type=str, default='weights/post_scale.pt', help='weights path')
    parser.add_argument('--w_bits', type=int, default=6, help='w-bit')
    parser.add_argument('--b_bits', type=int, default=6, help='w-bit')
    return parser.parse_args()    

def export_parameter_npy(model,path,export_quantized=False,w_bits=6,b_bits=6):
    b_min = -(1 << (b_bits - 1))
    b_max = (1 << (b_bits - 1)) - 1
    w_min = -(1 << (w_bits - 1))
    w_max = (1 << (w_bits - 1)) - 1   
    path = pathlib.Path(path).absolute()
    if path.exists():
        shutil.rmtree(path, ignore_errors=False, onerror=None)        
    path.mkdir(parents=True, exist_ok=True)
    parameters = model.state_dict()
    if not export_quantized:
        for key in parameters:                              
            np.save(str(path/key), parameters[key].cpu().data.numpy())
    else:
        for key in parameters:                              
            layer_name = key.split('.')
            layer_name = '.'.join(layer_name[:3])
            if ('.weight' in key) and not 'shift' in key:
                weight = parameters[key] << parameters[layer_name + '.weight_shift']
                weight = weight.cpu().data.numpy()
                weight = np.round(weight)
                weight = np.clip(weight,w_min,w_max)
                np.save(str(path/key),weight)
                
            if ('.bias' in key) and not 'shift' in key:
                bias = parameters[key] << parameters[layer_name + '.bias_shift']
                bias = bias.cpu().data.numpy()
                bias = np.round(bias)
                bias = np.clip(bias,b_min,b_max)  
                np.save(str(path/key),bias)
                
            if ('.activation_shift' in key):
                np.save(str(path/key),parameters[key].data.numpy())

def convert(opt):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights = opt.weights
    # Initialize
    device='cpu'
    device = torch_utils.select_device(device)
    output_layers = [15,22]
    # Initialize model
    model = Darknet(opt.cfg, img_size, quantized=opt.quantized)
    post_scale_model = Darknet(opt.cfg, img_size, quantized=7)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        chkpt = torch.load(weights, map_location=device)
        model.load_state_dict(chkpt['model'])
    else:  # darknet format
        raise NotImplementedError("Quantize weighs first")
        

    parameters = model.state_dict()
    new_parameters = post_scale_model.state_dict()
    layer_nbr = 0
    prev_layer_nbr = 0
    for key in parameters:
        if 'scale' in key:
            new_layer_nbr = [int(s) for s in key.split('.') if s.isdigit()][0]
            if new_layer_nbr != layer_nbr:
                prev_layer_nbr = layer_nbr
                layer_nbr = new_layer_nbr
                prev_layer_name = layer_name
            layer_name = key.split('.')
            layer_name = '.'.join(layer_name[:3])
    
            scale = parameters[key].cpu().data.numpy()
            scale = np.int32(np.round(np.log2(scale+1e-10)))*(-1)
            if'weight_quantizer.scale' in key:
                new_parameters[layer_name+".weight_shift"].copy_(torch.tensor(scale[0]))
                new_parameters[layer_name+".activation_shift"].add_(scale[0])                    
            elif 'activation_quantizer.scale' in key:
                if layer_nbr == 0:
                    scale[0] = 8
                    new_parameters[layer_name+".activation_shift"].copy_(torch.tensor(scale[0]))
                else:
                    if layer_nbr in output_layers:
                        new_parameters[layer_name+".activation_shift"].copy_(torch.tensor(0))
                        rshift_name = '.'.join(layer_name.split('.')[:2]) +".result_shift"
                        new_parameters[rshift_name+".shift"].copy_(torch.tensor(scale[0]))
                    else:
                        new_parameters[layer_name+".activation_shift"].copy_(torch.tensor(scale[0]))
                    if not prev_layer_nbr in output_layers:
                        new_parameters[prev_layer_name+".activation_shift"].sub_(scale[0])
                        if new_parameters[prev_layer_name+".activation_shift"] < 4:
                            new_parameters[prev_layer_name+".weight_shift"].add_(torch.tensor(1.0))
                            new_parameters[prev_layer_name+".activation_shift"].add_(torch.tensor(1.0))                        
                        
                        print(new_parameters[prev_layer_name+".activation_shift"])
                new_parameters[layer_name+".bias_shift"].copy_(torch.tensor(scale[0]))
            else:
                raise Exception("Key error:"+ key)
        elif ('.weight' in key or '.bias' in key) and key in new_parameters.keys():
            new_parameters[key] = parameters[key].clone()
                
                
    print(new_parameters.keys())
    post_scale_model.load_state_dict(new_parameters, strict=False)   
    # Eval mode
    post_scale_model.to(device).eval()
    if opt.param_outpath != None:
        export_parameter_npy(post_scale_model,opt.param_outpath,export_quantized=True,w_bits=opt.w_bits,b_bits=opt.b_bits)
        
    
    if hasattr(post_scale_model, 'module'):
        model_temp = post_scale_model.module.state_dict()
    else:
        model_temp = post_scale_model.state_dict()    
    
    chkpt = {'epoch': chkpt['epoch'],
             'best_fitness': chkpt['best_fitness'],
             'training_results': chkpt['training_results'],
             'model': model_temp,
             'optimizer': chkpt['optimizer']}

    torch.save(chkpt, opt.out_weights)    


if __name__ == '__main__':
    opt = _create_parser()
    print(opt)

    with torch.no_grad():
        convert(opt)
