from utils.google_utils import *
from utils.parse_config import *
from utils.quantized.quantized_google import *
from utils.quantized.quantized_dorefa import *
from utils.quantized.quantized_ptq import *
from utils.quantized.quantized_ptq_cos import *
from utils.quantized.quantized_intuitus import *
import utils.quantized.quantized_intuitus_new as q_new
from utils.layers import *
from collections import OrderedDict
import copy
import pathlib
import shutil
import numpy as np

ONNX_EXPORT = False


# YOLO
def create_modules(module_defs, img_size, cfg, quantized, quantizer_output, a_bit=8, w_bit=8, FPGA=False, steps=0):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1
    layer_norm_list = []

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            if 'quant' in mdef.keys():
                quant = int(mdef['quant'])
            else:
                quant = 0
                
            if 'quant_act' in mdef.keys():
                quant_act = int(mdef['quant_act'])
            else:
                quant_act = 0                
                
                
 
            if 'fused' in mdef.keys() and int(mdef['fused']) != 0:
                fused = True
            else:
                fused = False               

            if 'normalize' in mdef.keys() and int(mdef['normalize']) != 0:
                normalize = True
            else:
                normalize = False 

            if 'freeze' in mdef.keys() and int(mdef['freeze']) != 0:
                freeze = True
            else:
                freeze = False      

            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            if quantized == 1:
                if FPGA:
                    modules.add_module('Conv2d', BNFold_QuantizedConv2d_For_FPGA(in_channels=output_filters[-1],
                                                                                 out_channels=filters,
                                                                                 kernel_size=kernel_size,
                                                                                 stride=int(mdef['stride']),
                                                                                 padding=pad,
                                                                                 groups=mdef[
                                                                                     'groups'] if 'groups' in mdef else 1,
                                                                                 bias=not bn,
                                                                                 a_bits=a_bit,
                                                                                 w_bits=w_bit,
                                                                                 bn=bn,
                                                                                 activate=mdef['activation'],
                                                                                 steps=steps,
                                                                                 quantizer_output=quantizer_output))
                else:
                    modules.add_module('Conv2d', QuantizedConv2d(in_channels=output_filters[-1],
                                                                 out_channels=filters,
                                                                 kernel_size=kernel_size,
                                                                 stride=int(mdef['stride']),
                                                                 padding=pad,
                                                                 groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                 bias=not bn,
                                                                 a_bits=a_bit,
                                                                 w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU(inplace=True))
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
            elif quantized == 2:
                if FPGA:
                    modules.add_module('Conv2d', BNFold_DorefaConv2d(in_channels=output_filters[-1],
                                                                     out_channels=filters,
                                                                     kernel_size=kernel_size,
                                                                     stride=int(mdef['stride']),
                                                                     padding=pad,
                                                                     groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                     bias=not bn,
                                                                     a_bits=a_bit,
                                                                     w_bits=w_bit,
                                                                     bn=bn,
                                                                     activate=mdef['activation'],
                                                                     steps=steps))
                else:
                    modules.add_module('Conv2d', DorefaConv2d(in_channels=output_filters[-1],
                                                              out_channels=filters,
                                                              kernel_size=kernel_size,
                                                              stride=int(mdef['stride']),
                                                              padding=pad,
                                                              groups=mdef['groups'] if 'groups' in mdef else 1,
                                                              bias=not bn,
                                                              a_bits=a_bit,
                                                              w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU())
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
            elif quantized == 3:
                if FPGA:
                    modules.add_module('Conv2d', BNFold_PTQuantizedConv2d_For_FPGA(in_channels=output_filters[-1],
                                                                                   out_channels=filters,
                                                                                   kernel_size=kernel_size,
                                                                                   stride=int(mdef['stride']),
                                                                                   padding=pad,
                                                                                   groups=mdef[
                                                                                       'groups'] if 'groups' in mdef else 1,
                                                                                   bias=not bn,
                                                                                   a_bits=a_bit,
                                                                                   w_bits=w_bit,
                                                                                   bn=bn,
                                                                                   activate=mdef['activation']))
                else:
                    modules.add_module('Conv2d', PTQuantizedConv2d(in_channels=output_filters[-1],
                                                                   out_channels=filters,
                                                                   kernel_size=kernel_size,
                                                                   stride=int(mdef['stride']),
                                                                   padding=pad,
                                                                   groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                   bias=not bn,
                                                                   a_bits=a_bit,
                                                                   w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU())
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
            elif quantized == 4:
                modules.add_module('Conv2d', BNFold_COSPTQuantizedConv2d_For_FPGA(in_channels=output_filters[-1],
                                                                                  out_channels=filters,
                                                                                  kernel_size=kernel_size,
                                                                                  stride=int(mdef['stride']),
                                                                                  padding=pad,
                                                                                  groups=mdef[
                                                                                      'groups'] if 'groups' in mdef else 1,
                                                                                  bias=not bn,
                                                                                  a_bits=a_bit,
                                                                                  w_bits=w_bit,
                                                                                  bn=bn,
                                                                                  activate=mdef['activation'],
                                                                                  quantizer_output = quantizer_output))
            elif quantized == 5:

                if quant > 1:
                    quantize_weights = True
                    clip_weights = False 
                elif quant != 0:
                    quantize_weights = False 
                    clip_weights = True
                else:
                    quantize_weights = False 
                    clip_weights = False    

                if quant_act > 1:
                    quantize_activations = True
                    clip_activations = False 
                elif quant_act != 0:
                    quantize_activations = False 
                    clip_activations = True
                else:
                    quantize_activations = False 
                    clip_activations = False   

                if quant_act > 2:                 
                    modules.add_module('Conv2d', FPGA_IntuitusConv2d(
                                                                in_channels=output_filters[-1],
                                                                out_channels=filters,
                                                                kernel_size=kernel_size,
                                                                stride=int(mdef['stride']),
                                                                padding=pad,
                                                                groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                bias=not bn,
                                                                fold_bn = fused,
                                                                quantize_weights = quantize_weights,
                                                                clip_weights = clip_weights,
                                                                normalize_weights = normalize,
                                                                quantize_activations=quantize_activations,
                                                                clip_activations=clip_activations,
                                                                freeze=freeze))
                else:                  
                    modules.add_module('Conv2d', IntuitusConv2d(in_channels=output_filters[-1],
                                                                out_channels=filters,
                                                                kernel_size=kernel_size,
                                                                stride=int(mdef['stride']),
                                                                padding=pad,
                                                                groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                bias=not bn,
                                                                fold_bn = fused,
                                                                quantize_weights = quantize_weights,
                                                                clip_weights = clip_weights,
                                                                normalize_weights = normalize,
                                                                quantize_activations=quantize_activations,
                                                                clip_activations=clip_activations,
                                                                freeze=freeze))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
                        

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU())
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
                    # if mdef['activation'] == 'linear':
                    #     modules.add_module('norm_multiplier', IntuitusNorm_Multiplier())
                
                # for i, layer in enumerate(modules):
                #     if isinstance(layer, IntuitusConv2d) or isinstance(layer, IntuitusNorm_Multiplier):
                #         if layer_norm_list != []:
                #             layer.register_factorized_norm_observer(layer_norm_list[-1],layer_norm_list[-1].get_factorized_norm_oberserver)
                #         layer_norm_list.append(layer)
                
                    
                
            elif quantized == 6:
                if FPGA:
                    modules.add_module('Conv2d', q_new.QuantizedConv2d_fpga(in_channels=output_filters[-1],
                                                                            out_channels=filters,
                                                                            kernel_size=kernel_size,
                                                                            stride=int(mdef['stride']),
                                                                            padding=pad,
                                                                            groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                            bias=not bn,
                                                                            a_bits=a_bit,
                                                                            w_bits=w_bit))
                    if bn:                 
                        raise NotImplementedError("Batchnorm not supported for FPGA. Fuse batchnorm first using quantization mode -1 (no quantization).")
                else:
                    modules.add_module('Conv2d', q_new.QuantizedConv2d(in_channels=output_filters[-1],
                                                                       out_channels=filters,
                                                                       kernel_size=kernel_size,
                                                                       stride=int(mdef['stride']),
                                                                       padding=pad,
                                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                       bias=not bn,
                                                                       a_bits=a_bit,
                                                                       w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                    # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                    # modules.add_module('activation', Swish())
                if mdef['activation'] == 'relu6':
                    modules.add_module('activation', ReLU6())
                if mdef['activation'] == 'h_swish':
                    modules.add_module('activation', HardSwish())
                if mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU(inplace=True))
                if mdef['activation'] == 'mish':
                    modules.add_module('activation', Mish())   
            elif quantized == 7:
                if FPGA:
                    modules.add_module('Conv2d', q_new.QuantizedConv2d_post_shift_fpga(in_channels=output_filters[-1],
                                                                            out_channels=filters,
                                                                            kernel_size=kernel_size,
                                                                            stride=int(mdef['stride']),
                                                                            padding=pad,
                                                                            groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                            bias=not bn,
                                                                            a_bits=a_bit,
                                                                            w_bits=w_bit))
                    if bn:                 
                        raise NotImplementedError("Batchnorm not supported for FPGA. Fuse batchnorm first using quantization mode -1 (no quantization).")
                else:
                    modules.add_module('Conv2d', q_new.QuantizedConv2d_post_shift(in_channels=output_filters[-1],
                                                                       out_channels=filters,
                                                                       kernel_size=kernel_size,
                                                                       stride=int(mdef['stride']),
                                                                       padding=pad,
                                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                                       bias=not bn,
                                                                       a_bits=a_bit,
                                                                       w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                    # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                    # modules.add_module('activation', Swish())
                if mdef['activation'] == 'relu6':
                    modules.add_module('activation', ReLU6())
                if mdef['activation'] == 'h_swish':
                    modules.add_module('activation', HardSwish())
                if mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU(inplace=True))
                if mdef['activation'] == 'mish':
                    modules.add_module('activation', Mish())     
                if mdef['activation'] == 'linear':
                    modules.add_module('result_shift', q_new.IntuitusReslult_shift())   
            else:
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=kernel_size,
                                                       stride=int(mdef['stride']),
                                                       padding=pad,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                    # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                    # modules.add_module('activation', Swish())
                if mdef['activation'] == 'relu6':
                    modules.add_module('activation', ReLU6())
                if mdef['activation'] == 'h_swish':
                    modules.add_module('activation', HardSwish())
                if mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU())
                if mdef['activation'] == 'mish':
                    modules.add_module('activation', Mish())

        elif mdef['type'] == 'depthwise':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            if quantized == 1:
                if FPGA:
                    modules.add_module('DepthWise2d',
                                       BNFold_QuantizedConv2d_For_FPGA(in_channels=output_filters[-1],
                                                                       out_channels=filters,
                                                                       kernel_size=kernel_size,
                                                                       stride=int(mdef['stride']),
                                                                       padding=pad,
                                                                       groups=output_filters[-1],
                                                                       bias=not bn,
                                                                       a_bits=a_bit,
                                                                       w_bits=w_bit,
                                                                       bn=bn,
                                                                       activate=mdef['activation'],
                                                                       steps=steps,
                                                                       quantizer_output = quantizer_output))
                else:
                    modules.add_module('DepthWise2d', QuantizedConv2d(in_channels=output_filters[-1],
                                                                      out_channels=filters,
                                                                      kernel_size=kernel_size,
                                                                      stride=int(mdef['stride']),
                                                                      padding=pad,
                                                                      groups=output_filters[-1],
                                                                      bias=not bn,
                                                                      a_bits=a_bit,
                                                                      w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU())
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
            elif quantized == 2:
                if FPGA:
                    modules.add_module('DepthWise2d', BNFold_DorefaConv2d(in_channels=output_filters[-1],
                                                                          out_channels=filters,
                                                                          kernel_size=kernel_size,
                                                                          stride=int(mdef['stride']),
                                                                          padding=pad,
                                                                          groups=output_filters[-1],
                                                                          bias=not bn,
                                                                          a_bits=a_bit,
                                                                          w_bits=w_bit,
                                                                          bn=bn,
                                                                          activate=mdef['activation'],
                                                                          steps=steps))
                else:
                    modules.add_module('DepthWise2d', DorefaConv2d(in_channels=output_filters[-1],
                                                                   out_channels=filters,
                                                                   kernel_size=kernel_size,
                                                                   stride=int(mdef['stride']),
                                                                   padding=pad,
                                                                   groups=output_filters[-1],
                                                                   bias=not bn,
                                                                   a_bits=a_bit,
                                                                   w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU())
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
            elif quantized == 3:
                if FPGA:
                    modules.add_module('DepthWise2d', BNFold_PTQuantizedConv2d_For_FPGA(in_channels=output_filters[-1],
                                                                                        out_channels=filters,
                                                                                        kernel_size=kernel_size,
                                                                                        stride=int(mdef['stride']),
                                                                                        padding=pad,
                                                                                        groups=output_filters[-1],
                                                                                        bias=not bn,
                                                                                        a_bits=a_bit,
                                                                                        w_bits=w_bit,
                                                                                        bn=bn,
                                                                                        activate=mdef['activation']))
                else:
                    modules.add_module('DepthWise2d', PTQuantizedConv2d(in_channels=output_filters[-1],
                                                                        out_channels=filters,
                                                                        kernel_size=kernel_size,
                                                                        stride=int(mdef['stride']),
                                                                        padding=pad,
                                                                        groups=output_filters[-1],
                                                                        bias=not bn,
                                                                        a_bits=a_bit,
                                                                        w_bits=w_bit))
                    if bn:
                        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                    if mdef['activation'] == 'leaky':
                        modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                        # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                        # modules.add_module('activation', Swish())
                    if mdef['activation'] == 'relu6':
                        modules.add_module('activation', ReLU6())
                    if mdef['activation'] == 'h_swish':
                        modules.add_module('activation', HardSwish())
                    if mdef['activation'] == 'relu':
                        modules.add_module('activation', nn.ReLU())
                    if mdef['activation'] == 'mish':
                        modules.add_module('activation', Mish())
            elif quantized == 4:
                modules.add_module('DepthWise2d', BNFold_COSPTQuantizedConv2d_For_FPGA(in_channels=output_filters[-1],
                                                                                       out_channels=filters,
                                                                                       kernel_size=kernel_size,
                                                                                       stride=int(mdef['stride']),
                                                                                       padding=pad,
                                                                                       groups=output_filters[-1],
                                                                                       bias=not bn,
                                                                                       a_bits=a_bit,
                                                                                       w_bits=w_bit,
                                                                                       bn=bn,
                                                                                       activate=mdef['activation'],
                                                                                       quantizer_output = quantizer_output))
            else:
                modules.add_module('DepthWise2d', nn.Conv2d(in_channels=output_filters[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(mdef['stride']),
                                                            padding=pad,
                                                            groups=output_filters[-1],
                                                            bias=not bn), )
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                    # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                    # modules.add_module('activation', Swish())
                if mdef['activation'] == 'relu6':
                    modules.add_module('activation', ReLU6())
                if mdef['activation'] == 'h_swish':
                    modules.add_module('activation', HardSwish())
                if mdef['activation'] == 'relu':
                    modules.add_module('activation', nn.ReLU())
                if mdef['activation'] == 'mish':
                    modules.add_module('activation', Mish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'se':
            if 'filters' in mdef:
                filters = int(mdef['filters'])
                modules.add_module('se', SE(channel=filters))
            if 'reduction' in mdef:
                modules.add_module('se', SE(output_filters[-1], reduction=int(mdef['reduction'])))

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            if 'groups' in mdef:
                filters = filters // 2
            routs.extend([i + l if l < 0 else l for l in layers])
            if 'groups' in mdef:
                modules = FeatureConcat(layers=layers, groups=True)
            else:
                modules = FeatureConcat(layers=layers, groups=False)
                

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = Shortcut(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):  # stride order reversed
                if not 'yolov4-tiny' in cfg:
                    stride = list(reversed(stride))
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                with torch.no_grad():
                    j = layers[yolo_index] if 'from' in mdef else -1
                    bias_ = module_list[j][0].bias  # shape(255,)
                    bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                    bias[:, 4] = bias[:, 4] - 4.5  # obj
                    bias[:, 5:] = bias[:, 5:] + math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                    module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            # if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), verbose=False, quantized=-1, a_bit=8, w_bit=8, FPGA=False, quantizer_output=False, steps=0):
        super(Darknet, self).__init__()

        if isinstance(cfg, str):
            self.module_defs = parse_model_cfg(cfg)
        elif isinstance(cfg, list):
            self.module_defs = cfg
        self.quantized = quantized
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.FPGA = FPGA
        self.quantizer_output = quantizer_output####输出设置超参数 Hyperparameter für die Ausgabeeinstellung
        self.hyperparams = copy.deepcopy(self.module_defs[0])
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg, quantized=self.quantized,
                                                      quantizer_output=self.quantizer_output,a_bit=self.a_bit, w_bit=self.w_bit, FPGA=self.FPGA, steps=steps)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        #输出modelsummary
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False):

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out, feature_out = [], [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['Shortcut', 'FeatureConcat']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # Shortcut(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
                if name == "Sequential" and self.module_list[i + 1].__class__.__name__ != 'YOLOLayer':
                    feature_out.append(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return yolo_out, feature_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p, feature_out

    def fuse(self, quantized=-1, FPGA=False):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        if (quantized != -1 and quantized != 5) or FPGA == True:
            return
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        if quantized != 5:
            for a in list(self.children())[0]:
                if isinstance(a, nn.Sequential):
                    for i, b in enumerate(a):
                        if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                            # fuse this bn layer with the previous conv2d layer
                            conv = a[i - 1]
                            fused = torch_utils.fuse_conv_and_bn(conv, b, quantized, FPGA)
                            a = nn.Sequential(fused, *list(a.children())[i + 1:])
                            break
                fused_list.append(a)
        else:
            for a in list(self.children())[0]:
                if isinstance(a, nn.Sequential):
                    for i, b in enumerate(a):
                        if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                            # fuse this bn layer with the previous conv2d layer
                            conv = a[i - 1]
                            if isinstance(conv, IntuitusConv2d) and conv.fold_bn:
                                fused = torch_utils.fuse_conv_and_bn(conv, b, quantized, FPGA)
                                a = nn.Sequential(fused, *list(a.children())[i + 1:])
                            break
                fused_list.append(a)            
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def set_layer_norm(self, quantized=-1, FPGA=False):
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, IntuitusConv2d):
                        b.set_layer_norm()
                        break     

    def fuse_layer_norm(self):
        norm = torch.tensor(1.0,dtype=tc.float32)
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, IntuitusConv2d):
                        b.weight.data.copy_(b.weight*norm)
                        norm.data.copy_(b.layer_norm)
                        b.layer_norm.data.copy_(torch.tensor(1.0,dtype=tc.float32))
                        break    
                    print("Warning: module does not include an IntiutusConv2d Layer. Fuse layer norm could be faulty.")

    def set_quantized_weights(self):
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, IntuitusConv2d):
                        b.set_quantized_weights()
                        break    
                    print("Warning: module does not include an IntiutusConv2d Layer. Fuse layer norm could be faulty.")

    def write_parameter_to_txt(self,path,print_scale_as_txt = True):
        path = pathlib.Path(path).absolute()
        if path.exists():
            shutil.rmtree(path, ignore_errors=False, onerror=None)        
        path.mkdir(parents=True, exist_ok=True)
        parameters = self.state_dict()
        post_scale_dict = {}
        f=open(str(path/'scales.txt'),'a')
        layer_nbr = 0
        f.write("{}: ".format(layer_nbr))
        for key in parameters:
            new_layer_nbr = [int(s) for s in key.split('.') if s.isdigit()][0]
            if new_layer_nbr != layer_nbr:
                f.write("\n{}: ".format(new_layer_nbr))
                prev_layer_nbr = layer_nbr
                layer_nbr = new_layer_nbr
                
            if print_scale_as_txt and 'scale' in key:
                scale = parameters[key].cpu().data.numpy()
                scale = np.int32(np.round(np.log2(scale+1e-5)))
                if print_scale_as_txt and 'weight_quantizer.scale' in key:
                    f.write("w_scale: ")
                    post_scale_dict[str(layer_nbr)] += scale[0]
                elif print_scale_as_txt and 'activation_quantizer.scale' in key:
                    f.write("a_scale: ")    
                    if layer_nbr == 0:
                        scale[0] = -8
                        post_scale_dict[str(layer_nbr)] = scale[0]
                    else:
                        post_scale_dict[str(layer_nbr)] = scale[0]
                        post_scale_dict[str(prev_layer_nbr)] -= scale[0]
                else:
                    raise Exception("Key error:"+ key)
                np.savetxt(f, scale,'%.1f',delimiter=',')
                
            np.save(str(path/key), parameters[key].cpu().data.numpy())
        f.close()
        f=open(str(path/'post_scales.txt'),'a')
        print(post_scale_dict, file=f)
        f.close()
                
                
            

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1, pt=False, FPGA=False):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    quant = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            if mdef['batch_normalize']:
                if 'quant' in mdef.keys():
                    quant = int(mdef['quant'])
                else:
                    quant = 0
                    
                if FPGA and quant != 0:
                    # Load BN bias, weights, running mean and running variance
                    num_b = conv_layer.beta.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.beta)
                    conv_layer.beta.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.gamma)
                    conv_layer.gamma.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_mean)
                    conv_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_var)
                    conv_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                # elif quant == 2:
                #     num_b = conv_layer.out_channels  # Number of biases
                #     # Bias
                #     bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(torch.Tensor(num_b))
                #     ptr += num_b
                #     # Weight
                #     bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(torch.Tensor(num_b))
                #     ptr += num_b
                #     # Running Mean
                #     bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(torch.Tensor(num_b))
                #     bn_layer.running_mean.data.copy_(bn_rm)
                #     ptr += num_b
                #     # Running Var
                #     bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(torch.Tensor(num_b))
                #     bn_layer.running_var.data.copy_(bn_rv)
                #     ptr += num_b                    
                
                else:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    #if quant != 0:
                        #bn_layer.normalize_parameter() # normalize bn parameter to keep the results inside the quantization window
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
            else:
                # if os.path.basename(file) == 'yolov3.weights' or os.path.basename(file) == 'yolov3-tiny.weights':
                # pt标识使用coco预训练模型，读取参数时yolo层前面的一层输出为255
                if pt and os.path.basename(file).split('.')[-1] == 'weights':
                    num_b = 255
                    ptr += num_b
                    num_w = int(self.module_defs[i - 1]["filters"]) * 255
                    ptr += num_w
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
            if quant != 0:
                conv_layer.quantize_weights_and_bias() # start training from quantized weights and bias
        elif mdef['type'] == 'depthwise':
            depthwise_layer = module[0]
            if mdef['batch_normalize']:
                if FPGA:
                    # Load BN bias, weights, running mean and running variance
                    num_b = conv_layer.beta.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.beta)
                    conv_layer.beta.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.gamma)
                    conv_layer.gamma.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_mean)
                    conv_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.running_var)
                    conv_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
            # Load conv. weights
            num_w = depthwise_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(depthwise_layer.weight)
            depthwise_layer.weight.data.copy_(conv_w)
            ptr += num_w
        elif mdef['type'] == 'se':
            se_layer = module[0]
            fc = se_layer.fc
            fc1 = fc[0]
            num_fc1 = fc1.weight.numel()
            fc1_w = torch.from_numpy(weights[ptr:ptr + num_fc1]).view_as(fc1.weight)
            fc1.weight.data.copy_(fc1_w)
            ptr += num_fc1
            fc2 = fc[2]
            num_fc2 = fc2.weight.numel()
            fc2_w = torch.from_numpy(weights[ptr:ptr + num_fc2]).view_as(fc2.weight)
            fc2.weight.data.copy_(fc2_w)
            ptr += num_fc2

    # 确保指针到达权重的最后一个位置 Ensure that the pointer reaches the last position of the weight
    #assert ptr == len(weights)


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)
            elif mdef['type'] == 'depthwise':
                depthwise_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    depthwise_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                depthwise_layer.weight.data.cpu().numpy().tofile(f)
            elif mdef['type'] == 'se':
                se_layer = module[0]
                fc = se_layer.fc
                fc1 = fc[0]
                fc2 = fc[2]
                fc1.weight.data.cpu().numpy().tofile(f)
                fc2.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        target = weights.rsplit('.', 1)[0] + '.weights'
        save_weights(model, path=target, cutoff=-1)
        print("Success: converted '%s' to '%s'" % (weights, target))

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        target = weights.rsplit('.', 1)[0] + '.pt'
        torch.save(chkpt, target)
        print("Success: converted '%s' to '%'" % (weights, target))

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
