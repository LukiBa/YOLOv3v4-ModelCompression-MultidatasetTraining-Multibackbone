# Author:LiPu
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch as tc
import IntuitusExtension as C_impl


class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# ********************* Activation Quantization ***********************
class activation_quantize(nn.Module):
    def __init__(self, clip_only=False):
        super().__init__() 
        self.clip_only = clip_only
        self.eps = 1e-5

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.clip_only:
            return self.intuitus_clip_activation(input)     
        return self.intuiuts_quantize_activation(input)

    def intuiuts_quantize_activation(self,activation):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=activation.device)
        exp_width = tc.tensor(3.0,dtype=tc.float32,device=activation.device)     
        exp_zero_shift = tc.tensor(3.0,dtype=tc.float32,device=activation.device) 
        value = tc.abs(activation) + self.eps 
        exp = tc.floor(tc.clip((-1)*tc.log2(value)+exp_zero_shift,0,2.0**(exp_width)-1.0))
        mantissa = tc.round(activation*2.0**(exp-exp_zero_shift+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        return mantissa*2.0**(-exp+exp_zero_shift-mantissa_width)  

    def intuitus_clip_activation(self,activation):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=activation.device)
        exp_width = tc.tensor(3.0,dtype=tc.float32,device=activation.device)  
        exp_zero_shift = tc.tensor(4.0,dtype=tc.float32,device=activation.device)  
        q_min = 2.0**(-(2.0**exp_width-1.0)-mantissa_width+exp_zero_shift)
        q_max = 15.0*2.0**(-mantissa_width+exp_zero_shift)
        
        sign = activation.sign()
        activation_clip = tc.clamp(tc.abs(activation),q_min, q_max)
        activation_clip = tc.where(activation.abs()<q_min/2.0,tc.tensor(0.0,dtype=tc.float32,device=activation.device),activation_clip)
        activation_clip *= sign
        return activation_clip       

    def get_float8(self,activation):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=activation.device)
        exp_width = tc.tensor(3.0,dtype=tc.float32,device=activation.device)     
        exp_zero_shift = tc.tensor(4.0,dtype=tc.float32,device=activation.device) 
        value = tc.abs(activation) + self.eps 
        exp = tc.floor(tc.clip((-1)*tc.log2(value)+exp_zero_shift,0,2.0**(exp_width)-1.0))
        mantissa = tc.round(activation*2.0**(exp-exp_zero_shift+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        return exp.type(tc.int8), mantissa.type(tc.int8) 
    def to_float(self,np_exp,np_mantissa,device='cpu'):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=device)    
        exp_zero_shift = tc.tensor(3.0,dtype=tc.float32,device=device) 
        exp = tc.tensor(np_exp,dtype=tc.float32,device=device)    
        mantissa = tc.tensor(np_mantissa,dtype=tc.float32,device=device)         
        
        return mantissa*2.0**(-exp+exp_zero_shift-mantissa_width)        

# ********************* Weight Quantization ***********************
class weight_quantize(nn.Module):
    def __init__(self, clip_only=False):
        super().__init__()
        self.clip_only = clip_only
        self.eps = 1e-5
    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.clip_only:
            return self.intuitus_clip_weights(input)     
        return self.intuitus_quantize_weights(input)

    def get_weights(self, input):
        return self.intuitus_quantize_weights(input)
    
    def get_quantize_value(self,input):
        return self.intuitus_quantize_weights(input)
    
    def intuitus_quantize_weights(self,weights):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=weights.device)
        exp_width = tc.tensor(1.0,dtype=tc.float32,device=weights.device)        
        exp_shift = 2.0 + tc.round(tc.log2(tc.sqrt(tc.mean(weights**2,dim=(1,2,3))+self.eps)))
        exp_shift = tc.clip(exp_shift,tc.tensor(-2.0,dtype=tc.float32,device=weights.device),tc.tensor(1.0,dtype=tc.float32,device=weights.device))
        shift_multiplier = 2.0**((-1)*exp_shift.view(-1, 1, 1, 1))  
        weight_shift = weights*shift_multiplier
        value = tc.abs(weight_shift) + self.eps
        exp = tc.where(value==0.0,(2.0**exp_width)-1.0,(-1.0)*tc.log2(value))
        exp = tc.floor(tc.clip(exp,0.0,2.0**(exp_width)-1.0))                
        mantissa = tc.round(weight_shift*2.0**(exp+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        q_value = mantissa*2.0**(-exp-mantissa_width)
        return q_value, exp_shift      

    def intuitus_clip_weights(self,weights):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=weights.device)  
        q_min = 2.0**(-1.0-mantissa_width)
        q_max = 15.0*2.0**(-mantissa_width)
        exp_shift = 2.0 + tc.round(tc.log2(tc.sqrt(tc.mean(weights**2,dim=(1,2,3))+self.eps)))
        exp_shift = tc.clip(exp_shift,tc.tensor(-2.0,dtype=tc.float32,device=weights.device),tc.tensor(1.0,dtype=tc.float32,device=weights.device))
        shift_multiplier = 2.0**((-1)*exp_shift.view(-1, 1, 1, 1))  
        weight_shift = weights*shift_multiplier
        
        sign = weight_shift.sign()
        weight_clip = tc.clamp(tc.abs(weight_shift),q_min, q_max)
        weight_clip = tc.where(weight_shift.abs()<q_min/2.0,tc.tensor(0.0,dtype=tc.float32,device=weights.device),weight_clip)
        weight_clip *= sign
        return weight_clip, exp_shift   
    
    def get_float6(self,weights):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=weights.device)
        exp_width = tc.tensor(1.0,dtype=tc.float32,device=weights.device)        
        exp_shift = 2.0 + tc.round(tc.log2(tc.sqrt(tc.mean(weights**2,dim=(1,2,3))+self.eps)))
        exp_shift = tc.clip(exp_shift,tc.tensor(-2.0,dtype=tc.float32,device=weights.device),tc.tensor(1.0,dtype=tc.float32,device=weights.device))
        shift_multiplier = 2.0**((-1)*exp_shift.view(-1, 1, 1, 1))  
        weight_shift = weights*shift_multiplier
        value = tc.abs(weight_shift) + self.eps
        exp = tc.where(value==0.0,(2.0**exp_width)-1.0,(-1.0)*tc.log2(value))
        exp = tc.floor(tc.clip(exp,0.0,2.0**(exp_width)-1.0))                
        mantissa = tc.round(weight_shift*2.0**(exp+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        return exp.type(tc.int8), mantissa.type(tc.int8), exp_shift 

    
# ********************* Bias Quantization ***********************
class bias_quantize(nn.Module):
    def __init__(self, clip_only):
        super().__init__()
        self.clip_only = clip_only
        
    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input ,weight_exp_shift):
        if self.clip_only:
            return self.intuitus_clip_bias(input,weight_exp_shift)
        return self.intuitus_quantize_bias(input,weight_exp_shift)

    def get_bias(self, input ,weight_exp_shift):
        return self.intuitus_quantize_bias(input,weight_exp_shift)
    
    def get_quantize_value(self,input,weight_exp_shift):
        return self.intuitus_quantize_bias(input,weight_exp_shift)
    

    def intuitus_quantize_bias(self,bias,weight_exp_shift):
        bias_width = tc.tensor(4.0,dtype=tc.float32,device=bias.device)
        exponent =tc.tensor(3.0,dtype=tc.float32,device=bias.device)        
        bias_shift = bias*(2.0**((-1)*weight_exp_shift))
        bias_mantissa = tc.round(2.0**(exponent+bias_width-1.0)*bias_shift)
        bias_mantissa = tc.clip(bias_mantissa,(-1.0)*2.0**(bias_width-1.0),2.0**(bias_width-1.0)-1.0)
        return bias_mantissa*2.0**((-1.0)*exponent-bias_width+1.0)  

    def intuitus_clip_bias(self,bias,weight_exp_shift):
        bias_width = tc.tensor(4.0,dtype=tc.float32,device=bias.device)
        exponent =tc.tensor(3.0,dtype=tc.float32,device=bias.device)      
        q_min = 2.0**(-exponent-bias_width)
        q_max = 15.0*2.0**(-exponent-bias_width)
        
        bias_shift = bias*(2.0**((-1)*weight_exp_shift))
        
        sign = bias_shift.sign()
        bias_shift = bias_shift.abs_().clamp_(q_min, q_max)
        bias_shift *= sign
        return bias_shift 
    def get_fixed4(self,bias,weight_exp_shift):
        bias_width = tc.tensor(4.0,dtype=tc.float32,device=bias.device)
        exponent =tc.tensor(3.0,dtype=tc.float32,device=bias.device)        
        bias_shift = bias*(2.0**((-1)*weight_exp_shift))
        bias_mantissa = tc.round(2.0**(exponent+bias_width-1.0)*bias_shift)
        bias_mantissa = tc.clip(bias_mantissa,(-1.0)*2.0**(bias_width-1.0),2.0**(bias_width-1.0)-1.0)      
        return bias_mantissa.type(tc.int8)

class IntuitusConv2d(nn.Conv2d):
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
            normalize_weights=False,
            clip_weights=False,
            quantize_weights=False,
            clip_activations=False,
            quantize_activations=False,
            fold_bn = False,
            freeze = False,
            momentum = 0.1):
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
        # Define A and W quantizer
        self.activation_quantizer = activation_quantize(clip_only=clip_activations)
        self.weight_quantizer = weight_quantize(clip_only=clip_weights)
        if bias == True: 
            self.bias_quantizer = bias_quantize(clip_only=clip_weights)
        self.quantized = False
        self.normalized = not normalize_weights
        self.quantize_weights = quantize_weights or clip_weights
        self.quantize_activations = quantize_activations or clip_activations
        #self.layer_norm = tc.tensor(1.0,dtype=tc.float32)
        self.layer_norm_dyn = tc.tensor(1.0,dtype=tc.float32)
        self.weight_exp_shift = tc.zeros([out_channels], dtype=torch.float32)
        self.layer_norm = nn.Parameter(tc.tensor(1.0,dtype=tc.float32),requires_grad=False)
        # self.register_buffer('layer_norm_dyn', tc.tensor(1.0,dtype=tc.float32))
        # self.register_buffer('weight_exp_shift', tc.tensor(1.0,dtype=tc.float32))
        self.use_bias = bias
        self.fold_bn = fold_bn
        self.momentum = momentum
        self.running_mean = tc.tensor(0.0,dtype=tc.float32)
        self.running_var = tc.tensor(1.0,dtype=tc.float32)
        self.eps = 1e-5
        self.first_run = True 
        self.freeze = freeze
        self.clip_weights=clip_weights
        self.clip_activations=clip_activations
        # for param in self.parameters():
        #     param.requires_grad = not freeze

    def set_quantized_weights(self):
        if self.quantize_weights and not self.clip_weights:
            q_weight, weight_exp_shift = self.weight_quantizer(self.weight)
            if self.use_bias == True:
                q_bias = self.bias_quantizer(self.bias,weight_exp_shift)
            else:
                q_bias = self.bias
            self.weight.data.copy_(q_weight*2.0**(weight_exp_shift).view(-1, 1, 1, 1))
            self.bias.data.copy_(q_bias*2.0**(weight_exp_shift))

    def set_layer_norm(self):
        if not self.normalized:
            print("running mean: {}, running variance {}".format(self.running_mean,self.running_var))
            weight = self.weight/tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)
            bias = (self.bias - self.running_mean)/tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)
            self.bias.data.copy_(bias)
            self.weight.data.copy_(weight)
            self.layer_norm.data.copy_(self.layer_norm.mul(tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)))
            self.running_mean = tc.tensor(0.0,dtype=tc.float32)
            self.running_var = tc.tensor(0.3333333,dtype=tc.float32)
            
            self.first_run = False

    def forward(self, input):
        if self.weight_exp_shift.device != input.device:
            self.weight_exp_shift = self.weight_exp_shift.cuda()
            if self.weight_exp_shift.device != input.device:
                self.weight_exp_shift = self.weight_exp_shift.cpu()
                
        if not self.normalized and self.training:
            #self.normalize_weights_and_bias()
            for param in self.parameters():
                param.requires_grad = False
                
            output = F.conv2d(
                input=input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            batch_mean = torch.mean(output) # global mean
            batch_var = torch.var(output) # global var 

            with torch.no_grad():
                if self.first_run:
                    self.running_mean = batch_mean
                    self.running_var = batch_var
                    self.first_run = False
                else:
                    self.running_mean = self.running_mean + (batch_mean-self.running_mean)*self.momentum 
                    self.running_var = self.running_var + (batch_var-self.running_var)*self.momentum                 
                bias = self.bias
                weight = self.weight

        else:
            bias = self.bias
            weight = self.weight            
            
        if self.quantize_weights:
            q_weight, weight_exp_shift = self.weight_quantizer(weight)
            if self.use_bias == True:
                q_bias = self.bias_quantizer(bias,weight_exp_shift)
            else:
                q_bias = bias
            if self.clip_weights:
                self.weight.data.copy_(q_weight*2.0**(weight_exp_shift).view(-1, 1, 1, 1))
                self.bias.data.copy_(q_bias*2.0**(weight_exp_shift))
        else:
            q_weight = weight
            q_bias = bias
        
        output = F.conv2d(
            input=input,
            weight=q_weight,
            bias=q_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        output = output * self.layer_norm
        
        if self.quantize_activations:
            output = self.activation_quantizer(output)        
        
        if not self.quantize_weights:
            return output
        
        mult = 2.0**(weight_exp_shift)
        output = output * mult.view(1,-1,1,1) 
        if self.quantize_activations:
            output = self.activation_quantizer(output)          
        return output

def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


class FPGA_IntuitusConv2d(IntuitusConv2d):

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
            normalize_weights=False,
            clip_weights=False,
            quantize_weights=False,
            clip_activations=False,
            quantize_activations=False,
            fold_bn = False,
            freeze = False,
            momentum = 0.1):
        super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                normalize_weights=normalize_weights,
                clip_weights=clip_weights,
                quantize_weights=quantize_weights,
                clip_activations=clip_activations,
                quantize_activations=quantize_activations,
                fold_bn=fold_bn,
                freeze=freeze,
                momentum=momentum)

    def forward(self, input):
           
        w_exp, w_mant, weight_exp_shift = self.weight_quantizer.get_float6(self.weight)
        if self.use_bias == True:
            b_mant = self.bias_quantizer.get_fixed4(self.bias,weight_exp_shift)
        else:
            b_mant = tc.zeros(weight_exp_shift.shape,dtype=tc.int8)
            
        if self.stride[0] != self.stride[1]:
            raise NotImplementedError("Asymetric stride not supported yet.") 
        stride = int(self.stride[0])
        
        a_exp, a_mant = self.activation_quantizer.get_float8(input)        
        exp,mantissa =  C_impl.conv2d_fpga(a_exp.cpu().data.numpy(),a_mant.cpu().data.numpy(),
                                       w_exp.cpu().data.numpy(),w_mant.cpu().data.numpy(), 
                                       6, b_mant.cpu().data.numpy(),
                                       weight_exp_shift.type(tc.int8).cpu().data.numpy(),
                                       stride,1)            
        
        output = self.activation_quantizer.to_float(exp,mantissa)
        
        q_weight, weight_exp_shift = self.weight_quantizer(self.weight)
        if self.use_bias == True:
            q_bias = self.bias_quantizer(self.bias,weight_exp_shift)
        else:
            q_bias = self.bias        
        compare = F.conv2d(
            input=input,
            weight=q_weight,
            bias=q_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )  
        compare = F.relu(compare, inplace=True)
    
        
        return output


class IntuitusLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)

    def forward(self, input):
        q_input = self.activation_quantizer(input)
        q_weight, self.weight_exp_shift = self.weight_quantizer(self.weight)
        output = F.linear(input=q_input, weight=q_weight, bias=self.bias)
        #output = output * (2.0**self.weight_exp_shift)
        return output
