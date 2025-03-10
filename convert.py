# Author:LiPu
import argparse
import pathlib
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def convert():
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights = opt.weights
    # Initialize
    device = torch_utils.select_device(opt.device)

    # Initialize model
    model = Darknet(opt.cfg, img_size, is_gray_scale=opt.gray_scale)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
    outpath = pathlib.Path(__file__).parent / opt.output
    outpath = outpath / (opt.cfg.split('/')[-1].replace('.cfg', '') + '.pt')
    torch.save(model,str(outpath))
    save_weights(model, path='weights/best.weights')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov4tiny/yolov4-tiny.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov4-tiny.weights', help='path to weights file')
    parser.add_argument('--output', type=str, default='./pt_models', help='output folder')  # output folder
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--gray_scale', action='store_true', help='gray scale trainning')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        convert()
