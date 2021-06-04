import math
import time
import numpy as np
import pandas as pd
import scipy.io as io
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* range_trackers *********************
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':  # A,min_max_shape=(1, 1, 1, 1),layer
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # W,min_max_shape=(N, 1, 1, 1),channel
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
        elif self.q_level == 'C1':  # W,min_max_shape=(N, 1, 1, 1),channel级
            max_val = torch.max(torch.max(torch.max(torch.abs(input), 3, keepdim=True)[0], 2, keepdim=True)[0], 0, keepdim=True)[0]
            min_val = -max_val
        self.update_range(min_val, max_val)


class GlobalRangeTracker(RangeTracker):  # W,min_max_shape=(N, 1, 1, 1),channel min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros(1))
            self.register_buffer('max_val', torch.zeros(1))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        elif self.q_level == 'C1':
            self.register_buffer('min_val', torch.zeros(1,out_channels, 1, 1))
            self.register_buffer('max_val', torch.zeros(1,out_channels, 1, 1))                
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))


class AveragedRangeTracker(RangeTracker):  # A,min_max_shape=(1, 1, 1, 1),layer running_min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros(1))
            self.register_buffer('max_val', torch.zeros(1))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        elif self.q_level == 'C1':
            self.register_buffer('min_val', torch.zeros(1,out_channels, 1, 1))
            self.register_buffer('max_val', torch.zeros(1,out_channels, 1, 1))            
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)


# ********************* quantizers *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker, out_channels, FPGA, sign=True, log2_scale = False):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.FPGA = FPGA
        self.sign = sign
        self.log2_scale = log2_scale
        if out_channels == -1:
            self.register_buffer('scale', torch.zeros(1))  
            self.register_buffer('zero_point', torch.zeros(1))  
        else:
            self.register_buffer('scale', torch.zeros(out_channels, 1, 1, 1)) 
            self.register_buffer('zero_point', torch.zeros(out_channels, 1, 1, 1)) 

    def update_params(self):
        raise NotImplementedError

    def quantize(self, input):
        output = input / self.get_scale() #+ self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output
    
    def clamp(self, input):
        if self.sign:
            min_val = torch.tensor(-(1 << (self.bits - 1)))
            max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        if not self.sign:
            min_val = torch.tensor(0)
            max_val = torch.tensor((1 << self.bits) - 1)
        output = torch.clamp(input, min_val, max_val)
        return output

    def dequantize(self, input):
        output = input * self.get_scale() #(input - self.zero_point) * self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if self.training == True:
                self.range_tracker(input)
                self.update_params()
            output = self.quantize(input)
            output = self.round(output)
            output = self.clamp(output) 
            output = self.dequantize(output) 
        return output

    def get_quantize_value(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            output = self.quantize(input)
            output = self.round(output)
            output = self.clamp(output)
        return output
################ Get the number of shifts corresponding to the quantization factor
    def get_scale(self):
        ############ Shift correction
        scale = self.scale
        if self.log2_scale:
            scale = 2 **scale.log2().round()
        return scale

class SymmetricQuantizer(Quantizer):

    def update_params(self):
        if self.sign:
            min_val = torch.tensor(-(1 << (self.bits - 1)))
            max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        else:
            min_val = torch.tensor(0)
            max_val = torch.tensor((1 << self.bits) - 1)

        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val)) 

        float_max = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val)) 
        floor_float_range = 2 ** float_max.log2().floor()
        ceil_float_range = 2 ** float_max.log2().ceil()
        if abs(ceil_float_range - float_max) < abs(floor_float_range - float_max):
            float_range = ceil_float_range
        else:
            float_range = floor_float_range
        self.scale = float_range / quantized_range  
        self.zero_point = torch.zeros_like(self.scale)


class SymmetricOutQuantizer(SymmetricQuantizer):
    def forward(self, input):
        if self.training == True:
            self.range_tracker(input)
            self.update_params()  
        output = self.quantize(input) 
        output = self.round(input)
        output = self.clamp(output)          
        return output   

class Bias_Quantizer(nn.Module):
    def __init__(self,a_bits,sign=True):
        super().__init__()
        self.a_bits = a_bits
        self.sign = sign
        
    def round(self, input):
        output = Round.apply(input)
        return output
    
    def clamp(self, input):
        if self.sign:
            min_val = torch.tensor(-(1 << (self.a_bits - 1)))
            max_val = torch.tensor((1 << (self.a_bits - 1)) - 1)
        if not self.sign:
            min_val = torch.tensor(0)
            max_val = torch.tensor((1 << self.a_bits) - 1)
        output = torch.clamp(input, min_val, max_val)
        return output     
    def forward(self,input):
        output = input << (self.a_bits)
        output = self.round(output)
        output = self.clamp(output)   
        output = output >> (self.a_bits)
        return output

class QuantizedConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            a_bits=8,
            w_bits=8,
            b_bits=6):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.a_bits = a_bits
        self.w_bits = w_bits  
        self.b_bits = b_bits
        
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits,
                                                       range_tracker=AveragedRangeTracker(q_level='L',out_channels=-1),
                                                       out_channels=-1, FPGA=False,log2_scale=True)            
            
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, 
                                                   range_tracker=GlobalRangeTracker(q_level='L',out_channels=-1),
                                                   out_channels=-1, FPGA=False,log2_scale=True)


    def forward(self, input):
        if self.training:
            q_bias = self.bias
            bias_shift = self.b_bits-self.bias.abs().max().log2().ceil()
            q_bias = q_bias << bias_shift
            q_bias = Round.apply(q_bias)  
            q_bias = q_bias >> bias_shift   
            
            if input.shape[1] != 3:
                input = self.activation_quantizer(input)            

            q_weight = self.weight_quantizer(self.weight)
    
            output = F.conv2d(
                input=input,
                weight=q_weight,
                bias=q_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            return output
        else:            
            q_bias = self.bias
            bias_shift = self.b_bits-self.bias.abs().max().log2().ceil()
            q_bias = q_bias << bias_shift
            q_bias = Round.apply(q_bias)  
            q_bias = q_bias >> bias_shift            
            
            if input.shape[1] != 3:
                input = self.activation_quantizer.quantize(input)
                input = self.activation_quantizer.round(input)
                input = self.activation_quantizer.clamp(input)
                
                q_bias = self.activation_quantizer.quantize(q_bias)         
            else:
                input = input * 255.0
                q_bias = q_bias * 256.0
                
            q_bias = self.weight_quantizer.quantize(q_bias)          
                
            q_weight = self.weight_quantizer.quantize(self.weight)
            q_weight = self.weight_quantizer.round(q_weight)
            q_weight = self.weight_quantizer.clamp(q_weight)   
            
            #input = input * 2**(-self.a_bits+1)
            #q_weight = q_weight * 2**(-self.w_bits+1)
            #q_bias = q_bias* 2**(-self.a_bits+1)* 2**(-self.w_bits+1)
            
            output = F.conv2d(
                input=input,
                weight=q_weight,
                bias=q_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            #output = output * 2**(self.a_bits-1)* 2**(self.w_bits-1)
            if output.abs().max().log2().ceil() > 15.0:
                print('Overflow')
            if input.shape[1] != 3:
                output = self.activation_quantizer.dequantize(output)
            else:
                output = output / 256.0
            output = self.weight_quantizer.dequantize(output)
            return output     
        
        

class QuantizedConv2d_fpga(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            a_bits=8,
            w_bits=8,
            b_bits=6,
            interchannel_width = 12,
            intrachannel_width = 15
            ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.a_bits = a_bits
        self.w_bits = w_bits    
        self.b_bits = b_bits
        self.interchannel_shift = intrachannel_width - interchannel_width
        self.interchannel_width = intrachannel_width
        self.intrachannel_width = intrachannel_width
        
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits,
                                                       range_tracker=AveragedRangeTracker(q_level='L',out_channels=-1),
                                                       out_channels=-1, FPGA=False,log2_scale=True)            
            
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, 
                                                   range_tracker=GlobalRangeTracker(q_level='L',out_channels=-1),
                                                   out_channels=-1, FPGA=False,log2_scale=True)


    def forward(self, input):          
        q_bias = self.bias
        bias_shift = self.b_bits-self.bias.abs().max().log2().ceil()
        q_bias = q_bias << bias_shift
        q_bias = Round.apply(q_bias)  
        q_bias = q_bias >> bias_shift         
        
        if input.shape[1] != 3:
            input = self.activation_quantizer.quantize(input)
            input = self.activation_quantizer.round(input)
            input = self.activation_quantizer.clamp(input)
            
            q_bias = self.activation_quantizer.quantize(q_bias)         
        else:
            input = input * 255.0
            q_bias = q_bias * 256.0
            
        q_bias = self.weight_quantizer.quantize(q_bias)          
            
        q_weight = self.weight_quantizer.quantize(self.weight)
        q_weight = self.weight_quantizer.round(q_weight)
        q_weight = self.weight_quantizer.clamp(q_weight)   
        
        for i in range(input.shape[1]):
            if i == 0:
                interim = F.conv2d(
                    input=input[:,i:i+1,...],
                    weight=q_weight[:,i:i+1,...],
                    bias=q_bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                output = interim
            else:
                interim = F.conv2d(
                    input=input[:,i:i+1,...],
                    weight=q_weight[:,i:i+1,...],
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )                    
                output += interim  
            # if output.abs().max().log2().ceil() > (self.intrachannel_width-1.0):
            #     print('Overflow')
            output = output >> self.interchannel_shift
            output = self.activation_quantizer.round(output)
            output = torch.clamp(output,-1.0*2.0**(self.interchannel_width-1.0), 2.0**(self.interchannel_width-1.0)-1.0)
            output = output << self.interchannel_shift
                
        
        #output = self._quantized_conv2d(input,q_weight,q_bias,stride=self.stride,padding=self.padding)
        if input.shape[1] != 3:
            output = self.activation_quantizer.dequantize(output)
        else:
            output = output / 256.0
        output = self.weight_quantizer.dequantize(output)
        return output

    def _quantized_conv2d(self,input,weight,bias,stride=(1,1),padding=(1,1)):
        if input.shape[1] != weight.shape[1]:
            raise IndexError("Dimension Missmatch of activation input channel number and weight input channel number")

        out_shape = list(input.shape)
        out_shape[1] = list(weight.shape)[0]
        if bias != None:
            if weight.shape[0] != bias.shape[0]:
                raise IndexError("Dimension Missmatch of weights and bias")
            output = torch.ones(out_shape,dtype=input.dtype,device=input.device) 
            output *= bias.view(1, -1, 1, 1)
            
        else:
            output = torch.zeros(out_shape,dtype=input.dtype,device=input.device) 
        
        if padding == (1,1):
            padding_h = int(weight.shape[-2]/2.0)
            padding_v = int(weight.shape[-1]/2.0)
            padding_t = int(padding_h - (stride[0]-1.0))
            padding_b = int(padding_h)
            padding_l = int(padding_v - (stride[1]-1.0))
            padding_r = int(padding_v)
            pad = (padding_l,padding_r,padding_t,padding_b)
            input = torch.nn.functional.pad(input,pad,mode='constant',value=0.0)
            
        o_h,o_w = output.shape[-2:]
        
        for c in range(input.shape[1]):
            for h in range(weight.shape[-2]):
                for w in range(weight.shape[-1]):
                    output += input[:,c:c+1,h:o_h+h,w:o_w+w]*weight[:,c,h,w].view(1,-1,1,1)
                    
            output = output >> self.interchannel_shift
            output = self.activation_quantizer.round(output)
            output = torch.clamp(output,-1.0*2.0**(self.interchannel_width-1.0), 2.0**(self.interchannel_width-1.0)-1.0)
            output = output << self.interchannel_shift                    
          
        return output

class QuantizedConv2d_post_shift(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            a_bits=8,
            w_bits=8,
            b_bits=6):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.a_bits = a_bits
        self.w_bits = w_bits  
        self.b_bits = b_bits
        
        self.register_buffer('activation_shift', torch.zeros(1))
        self.register_buffer('weight_shift', torch.zeros(1))
        self.register_buffer('bias_shift', torch.zeros(1))
        
        self.a_min_val = torch.tensor(-(1 << (self.a_bits - 1)), requires_grad=False)
        self.a_max_val = torch.tensor((1 << (self.a_bits - 1)) - 1, requires_grad=False)
        self.b_min_val = torch.tensor(-(1 << (self.b_bits - 1)), requires_grad=False)
        self.b_max_val = torch.tensor((1 << (self.b_bits - 1)) - 1, requires_grad=False)
        self.w_min_val = torch.tensor(-(1 << (self.w_bits - 1)+1), requires_grad=False)
        self.w_max_val = torch.tensor((1 << (self.w_bits - 1)-1), requires_grad=False)
        
    def round(self, input):
        output = Round.apply(input)
        return output
    
    def forward(self, input):  
        if input.shape[1] == 3:                
            input = input * 255.0
            
        q_bias = self.bias
        q_bias = q_bias << self.bias_shift
        q_bias = q_bias << self.weight_shift
        q_bias = q_bias >> self.activation_shift
        q_bias = self.round(q_bias)
        q_bias = torch.clamp(q_bias,self.b_min_val,self.b_max_val)  
        q_bias = q_bias << self.activation_shift        
            
        q_weight = self.weight << self.weight_shift
        q_weight = self.round(q_weight)
        q_weight = torch.clamp(q_weight, self.w_min_val, self.w_max_val)    
        
        output = F.conv2d(
            input=input,
            weight=q_weight,
            bias=q_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        if output.abs().max().log2().ceil() > 15.0:
            print('Overflow')

        output = output >> self.activation_shift
        output = output = self.round(output)
        output = torch.clamp(output, self.a_min_val, self.a_max_val)           
        return output     

class QuantizedConv2d_post_shift_fpga(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            a_bits=8,
            w_bits=8,
            b_bits=6,
            interchannel_width = 12,
            intrachannel_width = 15
            ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.a_bits = a_bits
        self.w_bits = w_bits    
        self.b_bits = b_bits
        self.interchannel_shift = intrachannel_width - interchannel_width
        self.interchannel_width = intrachannel_width
        self.intrachannel_width = intrachannel_width
        
        self.register_buffer('activation_shift', torch.zeros(1))
        self.register_buffer('weight_shift', torch.zeros(1))
        self.register_buffer('bias_shift', torch.zeros(1))
        
        self.a_min_val = torch.tensor(-(1 << (self.a_bits - 1)), requires_grad=False)
        self.a_max_val = torch.tensor((1 << (self.a_bits - 1)) - 1, requires_grad=False)
        self.b_min_val = torch.tensor(-(1 << (self.b_bits - 1)), requires_grad=False)
        self.b_max_val = torch.tensor((1 << (self.b_bits - 1)) - 1, requires_grad=False)        
        self.w_min_val = torch.tensor(-(1 << (self.w_bits - 1)), requires_grad=False)
        self.w_max_val = torch.tensor((1 << (self.w_bits - 1)) - 1, requires_grad=False)

    def round(self, input):
        output = Round.apply(input)
        return output
        
    def forward(self, input):   
        if input.shape[1] == 3:                
            input = input * 255.0           
        
        q_bias = self.bias
        q_bias = q_bias << self.bias_shift
        q_bias = q_bias << self.weight_shift
        q_bias = q_bias >> self.activation_shift
        q_bias = self.round(q_bias)
        q_bias = torch.clamp(q_bias,self.b_min_val,self.b_max_val)  
        q_bias = q_bias << self.activation_shift          
            
        q_weight = self.weight << self.weight_shift
        q_weight = Round.apply(q_weight)
        q_weight = torch.clamp(q_weight, self.w_min_val, self.w_max_val) 
        
        for i in range(input.shape[1]):
            if i == 0:
                interim = F.conv2d(
                    input=input[:,i:i+1,...],
                    weight=q_weight[:,i:i+1,...],
                    bias=q_bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                output = interim
            else:
                interim = F.conv2d(
                    input=input[:,i:i+1,...],
                    weight=q_weight[:,i:i+1,...],
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )                    
                output += interim  
            # if output.abs().max().log2().ceil() > (self.intrachannel_width-1.0):
            #     print('Overflow')
            output = output >> self.interchannel_shift
            output = self.round(output)
            output = torch.clamp(output,-1.0*2.0**(self.interchannel_width-1.0), 2.0**(self.interchannel_width-1.0)-1.0)
            output = output << self.interchannel_shift
                
        
        #output = self._quantized_conv2d(input,q_weight,q_bias,stride=self.stride,padding=self.padding)
        output = output >> self.activation_shift
        output = output = Round.apply(output)
        output = torch.clamp(output, self.a_min_val, self.a_max_val) 
        return output

    def _quantized_conv2d(self,input,weight,bias,stride=(1,1),padding=(1,1)):
        if input.shape[1] != weight.shape[1]:
            raise IndexError("Dimension Missmatch of activation input channel number and weight input channel number")

        out_shape = list(input.shape)
        out_shape[1] = list(weight.shape)[0]
        if bias != None:
            if weight.shape[0] != bias.shape[0]:
                raise IndexError("Dimension Missmatch of weights and bias")
            output = torch.ones(out_shape,dtype=input.dtype,device=input.device) 
            output *= bias.view(1, -1, 1, 1)
            
        else:
            output = torch.zeros(out_shape,dtype=input.dtype,device=input.device) 
        
        if padding == (1,1):
            padding_h = int(weight.shape[-2]/2.0)
            padding_v = int(weight.shape[-1]/2.0)
            padding_t = int(padding_h - (stride[0]-1.0))
            padding_b = int(padding_h)
            padding_l = int(padding_v - (stride[1]-1.0))
            padding_r = int(padding_v)
            pad = (padding_l,padding_r,padding_t,padding_b)
            input = torch.nn.functional.pad(input,pad,mode='constant',value=0.0)
            
        o_h,o_w = output.shape[-2:]
        
        for c in range(input.shape[1]):
            for h in range(weight.shape[-2]):
                for w in range(weight.shape[-1]):
                    output += input[:,c:c+1,h:o_h+h,w:o_w+w]*weight[:,c,h,w].view(1,-1,1,1)
                    
            output = output >> self.interchannel_shift
            output = self.activation_quantizer.round(output)
            output = torch.clamp(output,-1.0*2.0**(self.interchannel_width-1.0), 2.0**(self.interchannel_width-1.0)-1.0)
            output = output << self.interchannel_shift                    
          
        return output

def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)

class IntuitusReslult_shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.zeros(1))

    def forward(self, input):
        output = input >> self.shift
        return output
