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
    def __init__(self, a_bits):
        super().__init__() 

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        #return input
        return self.intuiuts_quantize_activation(input)

    def intuiuts_quantize_activation(self,activation):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=activation.device)
        exp_width = tc.tensor(3.0,dtype=tc.float32,device=activation.device)           
        #sign = tc.where(activation<0.0,1.0,0.0)
        value = tc.abs(activation)
        exp = tc.where(value==0.0,(2.0**exp_width)-1.0,(-1.0)*tc.log2(value))
        exp = tc.floor(tc.clip(exp,0.0,2.0**(exp_width)-1.0))                
        mantissa = tc.round(activation*2.0**(exp+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        #q_value = tc.where(sign==0.0,mantissa*2.0**((-1.0)*exp-mantissa_width),(-1.0)*mantissa*2.0**((-1.0)*exp-mantissa_width))
        return mantissa*2.0**((-1.0)*exp-mantissa_width)  

# ********************* Weight Quantization ***********************
class weight_quantize(nn.Module):
    def __init__(self, w_bits):
        super().__init__()
        
    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        return self.intuitus_quantize_weights(input)

    def get_weights(self, input):
        return self.intuitus_quantize_weights(input)
    
    def get_quantize_value(self,input):
        return self.intuitus_quantize_weights(input)
    
    def intuitus_quantize_weights(self,weights):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=weights.device)
        exp_width = tc.tensor(1.0,dtype=tc.float32,device=weights.device)        
        self.exp_shift = 2.0 + tc.round(tc.log2(tc.sqrt(torch.abs(tc.mean(weights**2,dim=(1,2,3))))))
        shift_divider = 2.0**((-1)*self.exp_shift.view(-1, 1, 1, 1))
        weight_shift = weights*shift_divider
        #sign = tc.where(weights<0.0,1.0,0.0)
        value = tc.abs(weight_shift)
        exp = tc.where(value==0.0,(2.0**exp_width)-1.0,(-1.0)*tc.log2(value))
        exp = tc.floor(tc.clip(exp,0.0,2.0**(exp_width)-1.0))                
        mantissa = tc.round(weight_shift*2.0**(exp+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        #q_value = tc.where(sign==0.0,mantissa*2.0**((-1.0)*exp-mantissa_width),(-1.0)*mantissa*2.0**((-1.0)*exp-mantissa_width)) 
        q_value = mantissa*2.0**((-1.0)*exp-mantissa_width)
        return q_value, self.exp_shift      
    
# ********************* Bias Quantization ***********************
class bias_quantize(nn.Module):
    def __init__(self, b_bits):
        super().__init__()
        
    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input ,weight_exp_shift):
        bias_shift = input*(2.0**((-1)*weight_exp_shift))
        return self.intuiuts_quantize_activation(bias_shift)

    def get_bias(self, input ,weight_exp_shift):
        bias_shift = input*(2.0**((-1)*weight_exp_shift))
        return self.intuiuts_quantize_activation(bias_shift)        
        #return self.intuitus_quantize_bias(input,weight_exp_shift)
    
    def get_quantize_value(self,input,weight_exp_shift):
        bias_shift = input*(2.0**((-1)*weight_exp_shift))
        return self.intuiuts_quantize_activation(bias_shift)        
        #return self.intuitus_quantize_bias(input,weight_exp_shift)

    def intuitus_quantize_bias(self,bias,weight_exp_shift):
        bias_width = tc.tensor(4.0,dtype=tc.float32,device=bias.device)
        exponent =tc.tensor(0.0,dtype=tc.float32,device=bias.device)        
        bias_shift = bias*(2.0**((-1)*weight_exp_shift))
        bias_mantissa = tc.round(2.0**(exponent+bias_width-1.0)*bias_shift)
        bias_mantissa = tc.clip(bias_mantissa,(-1.0)*2.0**(bias_width-1.0),2.0**(bias_width-1.0)-1.0)
        return bias_mantissa*2.0**((-1.0)*exponent-bias_width+1.0)
    
    def intuiuts_quantize_activation(self,activation):
        mantissa_width = tc.tensor(4.0,dtype=tc.float32,device=activation.device)
        exp_width = tc.tensor(3.0,dtype=tc.float32,device=activation.device)           
        #sign = tc.where(activation<0.0,1.0,0.0)
        value = tc.abs(activation)
        exp = tc.where(value==0.0,(2.0**exp_width)-1.0,(-1.0)*tc.log2(value))
        exp = tc.floor(tc.clip(exp,0.0,2.0**(exp_width)-1.0))                
        mantissa = tc.round(activation*2.0**(exp+mantissa_width))
        mantissa = tc.clip(mantissa,(-1.0)*((2.0**mantissa_width)-1.0),(2.0**mantissa_width)-1.0)
        #q_value = tc.where(sign==0.0,mantissa*2.0**((-1.0)*exp-mantissa_width),(-1.0)*mantissa*2.0**((-1.0)*exp-mantissa_width))
        return mantissa*2.0**((-1.0)*exp-mantissa_width)      


# ********************* Quantized convlolution. Quantize W and Activation before convolution ***********************
class Intuitus_BachNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(Intuitus_BachNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.layer_norm = tc.tensor(1.0,dtype=tc.float32)
        self.bias_quantizer = bias_quantize(4)

    def normalize_parameter(self): # execute this function if new weights and biases are loaded --> do not execute during training --> may delete training progress
        # the goal is to keep the results inside the quantization window
        layer_norm = tc.sqrt(torch.abs(torch.var(self.weight)*2+tc.mean(self.weight)))+self.eps # eps used to avoid zero division error
        norm_weight = self.weight/layer_norm
        norm_bias = self.bias/layer_norm
        self.weight.data.copy_(norm_weight)
        q_bias = self.bias_quantizer(norm_bias,0)
        self.bias.data.copy_(q_bias)
        self.layer_norm.copy_(self.layer_norm*layer_norm)
        

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean.copy_(exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean)
                # update running_var with unbiased var
                self.running_var.copy_(exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var

        
        input = (input - mean[None, :, None, None]) / (torch.sqrt(torch.abs(var[None, :, None, None]*3)) + self.eps)
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


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
            quantize_weights=False,
            quantize_activations=False,
            fold_bn = False,
            momentum = 0.1,
            a_bits=8,
            w_bits=8,
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
        # Define A and W quantizer
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)
        if bias == True: 
            self.bias_quantizer = bias_quantize(b_bits=w_bits)
        self.quantized = False
        self.normalized = not normalize_weights
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.layer_norm = tc.tensor(1.0,dtype=tc.float32)
        self.layer_norm_dyn = tc.tensor(1.0,dtype=tc.float32)
        self.weight_exp_shift = tc.zeros([out_channels], dtype=torch.float32)
        # self.register_buffer('layer_norm', torch.zeros(out_channels))
        # self.register_buffer('layer_norm_dyn', tc.tensor(1.0,dtype=tc.float32))
        # self.register_buffer('weight_exp_shift', tc.tensor(1.0,dtype=tc.float32))
        self.use_bias = bias
        self.fold_bn = fold_bn
        self.momentum = momentum
        self.running_mean = tc.tensor(0.0,dtype=tc.float32)
        self.running_var = tc.tensor(1.0,dtype=tc.float32)
        self.eps = 1e-5
        self.first_run = True 

    def fuse_norm(self):
        if not self.normalized:
            print("running mean: {}, running variance {}".format(self.running_mean,self.running_var))
            weight = self.weight/tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)
            bias = (self.bias - self.running_mean)/tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)
            self.bias.data.copy_(bias)
            self.weight.data.copy_(weight)
            self.running_mean = tc.tensor(0.0,dtype=tc.float32)
            self.running_var = tc.tensor(0.3333333,dtype=tc.float32)
            self.first_run = False

    def normalize_weights_and_bias(self):
        layer_norm = (tc.sqrt(tc.abs(torch.var(self.weight)*2.0+tc.mean(self.weight)))*2.0)+self.eps # eps used to avoid zero division error
        norm_weight = self.weight/layer_norm
        self.weight.data.copy_(norm_weight)
        if self.bias != None:
            norm_bias = self.bias/layer_norm
            self.bias.data.copy_(norm_bias)
        
        self.layer_norm.copy_(self.layer_norm*layer_norm)
        
    def quantize_weights_and_bias(self): # execute this function if new weights and biases are loaded --> do not execute during training --> may delete training progress
        if self.normalized == False:
            self.normalize_weights_and_bias()
        
        q_weight, weight_exp_shift = self.weight_quantizer(self.weight)
        self.weight_exp_shift.copy_(weight_exp_shift)
        self.weight.data.copy_(q_weight)
        if self.use_bias == True:
            q_bias = self.bias_quantizer(self.bias,weight_exp_shift) 
            self.bias.data.copy_(q_bias)

    def forward(self, input):
        if self.weight_exp_shift.device != input.device:
            self.weight_exp_shift = self.weight_exp_shift.cuda()
            if self.weight_exp_shift.device != input.device:
                self.weight_exp_shift = self.weight_exp_shift.cpu()
                
        if not self.normalized and self.training:
            #self.normalize_weights_and_bias()

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
                weight = self.weight/tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)
                bias = (self.bias - self.running_mean)/tc.sqrt(tc.abs(self.running_var)*3.0 + self.eps)


        else:
            bias = self.bias
            weight = self.weight            
            
            
            
            
        
        if input.shape[1] != 3 and self.quantize_activations:
            input = self.activation_quantizer(input)
            
        if self.quantize_weights:
            q_weight, weight_exp_shift = self.weight_quantizer(weight)
            if self.use_bias == True:
                q_bias = self.bias_quantizer(bias,weight_exp_shift)
            else:
                q_bias = bias
            
            mean_shift = tc.floor(tc.mean(self.weight_exp_shift+weight_exp_shift))
            self.layer_norm_dyn = 2.0**(tc.floor(tc.mean(self.weight_exp_shift+weight_exp_shift)))
            weight_exp_shift -= mean_shift
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
        if not self.quantize_weights:
            return output
        
        mult = 2.0**(self.weight_exp_shift+weight_exp_shift)
        output = output * mult.view(1,-1,1,1) 
        return output

def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


class BNFold_IntuitusConv2d(IntuitusConv2d):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            eps=1e-5,
            momentum=0.01,  # 考虑量化带来的抖动影响,对momentum进行调整(0.1 ——> 0.01),削弱batch统计参数占比，一定程度抑制抖动。经实验量化训练效果更好,acc提升1%左右
            a_bits=8,
            w_bits=8,
            bn=0,
            activate='leaky',
            steps=0,
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
        self.bn = bn
        self.activate = activate
        self.eps = eps
        self.momentum = momentum
        self.freeze_step = int(steps * 0.9)
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.zeros(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        self.register_buffer('step', torch.zeros(1))

        init.normal_(self.gamma, 1, 0.5)
        init.zeros_(self.beta)

        # 实例化量化器（A-layer级，W-channel级） Quantisierer instanziieren
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)
        self.bias_quantizer = bias_quantize(b_bits=w_bits)

    def forward(self, input):
        # 训练态
        if self.training:
            self.step += 1
            if self.bn:
                # 先做普通卷积得到A，以取得BN参数 First do ordinary convolution to get A to get BN parameters
                output = F.conv2d(
                    input=input,
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                # 更新BN统计参数（batch和running）
                dims = [dim for dim in range(4) if dim != 1]
                self.batch_mean = torch.mean(output, dim=dims)
                self.batch_var = torch.var(output, dim=dims)

                with torch.no_grad():
                    if self.first_bn == 0 and torch.equal(self.running_mean, torch.zeros_like(
                            self.running_mean)) and torch.equal(self.running_var, torch.zeros_like(self.running_var)):
                        self.first_bn.add_(1)
                        self.running_mean.add_(self.batch_mean)
                        self.running_var.add_(self.batch_var)
                    else:
                        self.running_mean.mul_(1 - self.momentum).add_(self.momentum * self.batch_mean)
                        self.running_var.mul_(1 - self.momentum).add_(self.momentum * self.batch_var)
                # BN融合 BN fold
                if self.step < self.freeze_step:
                    if self.bias is not None:
                        bias = reshape_to_bias(
                            self.beta + (self.bias - self.batch_mean) * (
                                    self.gamma / torch.sqrt(tc.abs(self.batch_var + self.eps))))
                    else:
                        bias = reshape_to_bias(
                            self.beta - self.batch_mean * (
                                    self.gamma / torch.sqrt(tc.abs(self.batch_var + self.eps))))  # b融batch
                    weight = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(torch.abs(self.batch_var + self.eps)))  # w融running
                else:
                    if self.bias is not None:
                        bias = reshape_to_bias(
                            self.beta + (self.bias - self.running_mean) * (
                                    self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps))))
                    else:
                        bias = reshape_to_bias(
                            self.beta - self.running_mean * (
                                    self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps))))  # b融batch
                    weight = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps))) # w融running

            else:
                bias = self.bias
                weight = self.weight
        # 测试态
        else:
            # print(self.running_mean, self.running_var)
            # BN融合
            if self.bn:
                if self.bias is not None:
                    bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                            self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps))))
                else:
                    bias = reshape_to_bias(
                        self.beta - self.running_mean * (
                                self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps))))  # b融running
                weight = self.weight * reshape_to_weight(
                    self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps)))  # w融running
            else:
                bias = self.bias
                weight = self.weight
        # 量化A和bn融合后的W
        q_weight, weight_exp_shift = self.weight_quantizer(self.weight)
        self.weight_exp_shift = weight_exp_shift
        q_bias = self.bias_quantizer(bias,0)
        # 量化卷积
        if self.training:  # 训练态
            output = F.conv2d(
                input=input,
                weight=q_weight,
                # bias=self.bias,  # 注意，这里不加bias（self.bias为None）
                bias=q_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )

        else:  # 测试态
            output = F.conv2d(
                input=input,
                weight=q_weight,
                bias=q_bias,  # 注意，这里加bias，做完整的conv+bn
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        mult = (2.0**self.weight_exp_shift.view(1,-1,1,1))
        output = output * mult 
        if self.activate == 'leaky':
            output = F.leaky_relu(output, 0.125, inplace=True)
        elif self.activate == 'relu6':
            output = F.relu6(output, inplace=True)
        elif self.activate == 'h_swish':
            output = output * (F.relu6(output + 3.0, inplace=True) / 6.0)
        elif self.activate == 'relu':
            output = F.relu(output, inplace=True)
        elif self.activate == 'mish':
            output = output * F.softplus(output).tanh()
        elif self.activate == 'linear':
            return output
            # pass
        else:
            print(self.activate + " is not supported !")
        output = self.activation_quantizer(output)
        return output

    def BN_fuse(self):
        if self.bn:
            # BN融合
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                        self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps))))
            else:
                bias = reshape_to_bias(
                    self.beta - self.running_mean * self.gamma / torch.sqrt(
                        self.running_var + self.eps))  # b融running
            weight = self.weight * reshape_to_weight(
                self.gamma / torch.sqrt(torch.abs(self.running_var + self.eps)))  # w融running
        else:
            bias = self.bias
            weight = self.weight
        return weight, bias


class IntuitusLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.activation_quantizer = activation_quantize(a_bits=a_bits)
        self.weight_quantizer = weight_quantize(w_bits=w_bits)

    def forward(self, input):
        # 量化A和W
        q_input = self.activation_quantizer(input)
        q_weight, self.weight_exp_shift = self.weight_quantizer(self.weight)
        # 量化全连接
        output = F.linear(input=q_input, weight=q_weight, bias=self.bias)
        #output = output * (2.0**self.weight_exp_shift)
        return output
