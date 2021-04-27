import argparse
import json
from pathlib import Path
import shutil
from torch.utils.data import DataLoader
from models import *
from utils.datasets import *
from utils.utils import *
import numpy as np


def _create_parser():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--model', type=str, default=None, help='model path') #'pt_models/dorefa.pt'
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3tiny/yolov3-tiny-quant.cfg', help='*.cfg path')
    parser.add_argument('--cfg_gnd', type=str, default='./cfg/yolov3tiny/yolov3-tiny-fused.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='weights path')
    parser.add_argument('--gnd_weights', type=str, default='weights/last_v3_ql2_0.pt', help='gnd weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--img_outpath', type=str, default='./detect_imgs', help='Path to output images. If set to None images are not saved.') #'./detect_imgs'
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--quantized', type=int, default=5,
                        help='0:quantization way one Ternarized weight and 8bit activation')
    parser.add_argument('--a-bit', type=int, default=8,
                        help='a-bit')
    parser.add_argument('--w-bit', type=int, default=8,
                        help='w-bit')
    parser.add_argument('--FPGA', type=bool, default=False, help='FPGA')
    return parser.parse_args()    

def test(cfg,
         cfg_gnd,
         data,
         weights=None,
         gnd_weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=True,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True,
         quantized=-1,
         a_bit=8,
         w_bit=8,
         FPGA=False,
         rank=None,
         img_outpath=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz, quantized=quantized, a_bit=a_bit, w_bit=w_bit,
                        FPGA=FPGA)
        model_gnd = Darknet(cfg_gnd, imgsz, quantized=quantized, a_bit=a_bit, w_bit=w_bit,
                        FPGA=FPGA)
        
        # Load weights
        #attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
            model_gnd.load_state_dict(torch.load(gnd_weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights, FPGA=FPGA)
            load_darknet_weights(model_gnd, gnd_weights, FPGA=FPGA)

        # Fuse
        model.fuse(quantized=quantized, FPGA=opt.FPGA)
        model.to(device)
        print(model)
        model_gnd.fuse(quantized=quantized, FPGA=opt.FPGA)
        model_gnd.to(device)  
        print(model_gnd)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        # summary(model, input_size=(3, imgsz, imgsz))
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False
    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()
    
    if img_outpath != None:
        img_outpath = Path(img_outpath).absolute()
        img_predpath = img_outpath 
        img_gndpath = img_outpath 
        if img_predpath.exists():
            shutil.rmtree(img_predpath, ignore_errors=False, onerror=None)
        if img_gndpath.exists():
            shutil.rmtree(img_predpath, ignore_errors=False, onerror=None)    
        img_predpath.mkdir(parents=True, exist_ok=True)
        img_gndpath.mkdir(parents=True, exist_ok=True)
        
    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels2(path, imgsz, batch_size, single_cls=single_cls)
        test = dataset[0]
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=1,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    model_gnd.eval()
    # _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    #pbar = tqdm(dataloader, desc=s) if rank in [-1, 0] else dataloader
    pbar = tqdm(dataloader, desc=s)
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()

            inf_out, train_out, layer_outs = model(imgs, augment=augment)  # inference and training outputs
            inf_out_gnd, train_out_gnd, layer_outs_gnd = model_gnd(imgs, augment=augment)
            
            param_test = {}
            for name, param in model.named_parameters():
                param_test[name] = param
            
            param_gnd = {}
            for name, param in model_gnd.named_parameters():
                param_gnd[name] = param
                
            return layer_outs, layer_outs_gnd, param_test, param_gnd
            

if __name__ == '__main__':
   
    opt = _create_parser()     
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.data = list(glob.iglob('./**/' + opt.data, recursive=True))[0]  # find file
    #opt.FPGA = True
    model = None
    if opt.model != None:
        model = torch.load(opt.model)

    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test, gnd, param_test, param_gnd = test(opt.cfg,
                                                 opt.cfg_gnd,
                                                 opt.data,
                                                 opt.weights,
                                                 opt.gnd_weights,
                                                 opt.batch_size,
                                                 opt.img_size,
                                                 opt.conf_thres,
                                                 opt.iou_thres,
                                                 opt.save_json,
                                                 opt.single_cls,
                                                 opt.augment,
                                                 model = model,
                                                 quantized=opt.quantized,
                                                 a_bit=opt.a_bit,
                                                 w_bit=opt.w_bit,
                                                 FPGA=opt.FPGA,
                                                 img_outpath = opt.img_outpath)
        
        layer_out = -1
        go = gnd[layer_out]
        g = go.cpu().data.numpy()[0,...]
        out = test[layer_out]
        o = out.cpu().data.numpy()[0,...]
        t = o-g
        print(np.max(t))
        #w_gnd = param_gnd['module_list.'+str(layer_out)+'.Conv2d.weight'].cpu().data.numpy()
        #w_test = param_test['module_list.'+str(layer_out)+'.Conv2d.weight'].cpu().data.numpy()
        #wt = w_test[:,0,...]
        #wg = w_gnd[:,0,...]

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')
