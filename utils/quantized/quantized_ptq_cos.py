# Author:LiPu
import math
import time
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output


class Quantizer(nn.Module):
    def __init__(self, bits, out_channels):
        super().__init__()
        self.bits = bits
        if out_channels == -1:
            self.register_buffer('scale', torch.zeros(1))  # 量化比例因子
            self.register_buffer('float_range', torch.zeros(1))
        else:
            self.register_buffer('scale', torch.zeros(out_channels, 1, 1, 1))  # 量化比例因子
            self.register_buffer('float_range', torch.zeros(out_channels, 1, 1, 1))
        self.scale_list = [0 for i in range(bits)]

    def update_params(self, step):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        quantized_range = torch.max(torch.abs(min_val), torch.abs(max_val))  # 量化后范围
        temp = self.float_range
        self.float_range.add_(-temp).add_(2 ** step)
        self.scale = self.float_range / quantized_range  # 量化比例因子

    # 量化
    def quantize(self, input):
        output = input / self.scale
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # 截断
    def clamp(self, input):
        min_val = torch.tensor(-(1 << (self.bits - 1)))
        max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        output = torch.clamp(input, min_val, max_val)
        return output

    # 反量化
    def dequantize(self, input):
        output = (input) * self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if self.training == True:
                max_metrics = -1
                max_step = 0
                for i in range(self.bits):
                    self.update_params(i)
                    output = self.quantize(input)  # 量化
                    output = self.round(output)
                    output = self.clamp(output)  # 截断
                    output = self.dequantize(output)  # 反量化
                    cosine_similarity = torch.cosine_similarity(input.view(-1), output.view(-1), dim=0)
                    if cosine_similarity > max_metrics:
                        max_metrics = cosine_similarity
                        max_step = i
                self.scale_list[max_step] += 1
                Global_max_step = self.scale_list.index(max(self.scale_list))
                self.update_params(Global_max_step)

            output = self.quantize(input)  # 量化
            output = self.round(output)
            output = self.clamp(output)  # 截断
            output = self.dequantize(output)  # 反量化
            return output

    def get_quantize_value(self, input):

        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            output = self.quantize(input)  # 量化
            output = self.round(output)
            output = self.clamp(output)  # 截断
        return output


    ################获得量化因子所对应的移位数
    def get_scale(self):
        #############移位修正
        move_scale = math.log2(self.scale)
        move_scale = np.array(move_scale).reshape(1, -1)
        return move_scale


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


# ********************* bn融合_量化卷积（bn融合后，同时量化A/W，并做卷积） *********************


class BNFold_COSPTQuantizedConv2d_For_FPGA(nn.Conv2d):
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
            quantizer_output=False,
            reorder=False, TM=32, TN=32
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = bn
        self.activate = activate
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.register_buffer('batch_mean', torch.zeros(out_channels))
        self.register_buffer('batch_var', torch.zeros(out_channels))
        self.quantizer_output = quantizer_output
        self.reorder = reorder
        self.TM = TM
        self.TN = TN

        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = Quantizer(bits=a_bits, out_channels=-1)
        self.weight_quantizer = Quantizer(bits=w_bits, out_channels=-1)
        self.bias_quantizer = Quantizer(bits=w_bits, out_channels=-1)

    def forward(self, input):
        if self.bn:
            # BN融合
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                        self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(
                    self.beta - self.running_mean * self.gamma / torch.sqrt(
                        self.running_var + self.eps))  # b融running
            weight = self.weight * reshape_to_weight(
                self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
        else:
            bias = self.bias
            weight = self.weight

        # 量化A和bn融合后的W
        q_weight = self.weight_quantizer(weight)
        q_bias = self.bias_quantizer(bias)

        if self.quantizer_output == True:  # 输出量化参数txt文档

            # 创建的quantizer_output输出文件夹
            if not os.path.isdir('./quantizer_output'):
                os.makedirs('./quantizer_output')

            if not os.path.isdir('./quantizer_output/q_weight_out'):
                os.makedirs('./quantizer_output/q_weight_out')
            if not os.path.isdir('./quantizer_output/w_scale_out'):
                os.makedirs('./quantizer_output/w_scale_out')
            if not os.path.isdir('./quantizer_output/q_weight_max'):
                os.makedirs('./quantizer_output/q_weight_max')
            if not os.path.isdir('./quantizer_output/max_weight_count'):
                os.makedirs('./quantizer_output/max_weight_count')

            if not os.path.isdir('./quantizer_output/q_weight_reorder'):
                os.makedirs('./quantizer_output/q_weight_reorder')
            if not os.path.isdir('./quantizer_output/q_bias_reorder'):
                os.makedirs('./quantizer_output/q_bias_reorder')

            #######################输出当前层的权重量化因子
            weight_scale = self.weight_quantizer.get_scale()
            np.savetxt(('./quantizer_output/w_scale_out/%f.txt' % time.time()), weight_scale, delimiter='\n')
            #######################输出当前层的量化权重
            q_weight_txt = self.weight_quantizer.get_quantize_value(weight)

            #############权重重排序

            w_para = q_weight_txt  # 重排序参数
            if self.reorder == True:
                #print("use weights reorder!")
                shape_output = w_para.shape[0]
                shape_input = w_para.shape[1]
                num_TN = int(shape_input / self.TN)
                remainder_TN = shape_input % self.TN
                num_TM = int(shape_output / self.TM)
                remainder_TM = shape_output % self.TM
                first = True
                reorder_w_para = None
                if self.activate == 'linear':
                    print('layer-linear reorder!')
                    for k in range(num_TN):
                        temp = w_para[0:remainder_TM, k * self.TN:(k + 1) * self.TN, :, :]
                        temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                        temp = temp.permute(2, 0, 1).contiguous().view(-1)
                        if first:
                            reorder_w_para = temp.clone().cpu().data.numpy()
                            first = False
                        else:
                            reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())
                else:
                    for j in range(num_TM):
                        if shape_input == 3 or shape_input == 1:  # 第一层
                            print('The first layer~~~~~~~~~~~~')
                            temp = w_para[j * self.TM:(j + 1) * self.TM,
                                   num_TN * self.TN:num_TN * self.TN + remainder_TN, :,
                                   :]
                            temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                            fill = torch.zeros(self.TM, self.TN, temp.shape[2]).to(temp.device)
                            fill[:, 0:remainder_TN, :] = temp
                            temp = fill.permute(2, 0, 1).contiguous().view(-1)
                            if first:  # 创建数组存储
                                reorder_w_para = temp.clone().cpu().data.numpy()
                                first = False
                            else:
                                reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())
                        else:
                            for k in range(num_TN):
                                temp = w_para[j * self.TM:(j + 1) * self.TM, k * self.TN:(k + 1) * self.TN, :, :]
                                # #合并成论文图10(a)的TM*TN*(K2)的张量格式
                                temp = temp.view(temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3])
                                # 转换为图10(b)的重排序格式
                                temp = temp.permute(2, 0, 1).contiguous().view(-1)
                                if first:
                                    reorder_w_para = temp.clone().cpu().data.numpy()
                                    first = False
                                else:
                                    reorder_w_para = np.append(reorder_w_para, temp.cpu().data.numpy())

                w_para_flatten = reorder_w_para
                # print(reorder_w_para.size)
                #####验证重排序结果的正确性
                '''if w_para_flatten.size == w_para.shape[0] * w_para.shape[1] * w_para.shape[2] * w_para.shape[3]:
                    print("weights convert correctly!")
                else:
                    print("weights convert mismatchingly!")'''

                q_weight_reorder = w_para_flatten
                q_weight_reorder = np.array(q_weight_reorder).reshape(1, -1)
                np.savetxt(('./quantizer_output/q_weight_reorder/%f.txt' % time.time()), q_weight_reorder,
                           delimiter='\n')
            ################权重重排序结束

            q_weight_txt = np.array(q_weight_txt.cpu()).reshape(1, -1)
            q_weight_max = [np.max(q_weight_txt)]
            # q_weight_max = np.argmax(q_weight_txt)
            max_weight_count = [np.sum(abs(q_weight_txt) >= 127)]  # 统计该层溢出的数目
            np.savetxt(('./quantizer_output/max_weight_count/%f.txt' % time.time()), max_weight_count)
            np.savetxt(('./quantizer_output/q_weight_max/%f.txt' % time.time()), q_weight_max)
            np.savetxt(('./quantizer_output/q_weight_out/%f.txt' % time.time()), q_weight_txt, delimiter='\n')
            # io.savemat('save.mat',{'q_weight_txt':q_weight_txt})

            #######################创建输出偏置txt的文件夹
            if not os.path.isdir('./quantizer_output/q_bias_out'):
                os.makedirs('./quantizer_output/q_bias_out')
            if not os.path.isdir('./quantizer_output/b_scale_out'):
                os.makedirs('./quantizer_output/b_scale_out')
            #######################输出当前层偏置的量化因子
            bias_scale = self.bias_quantizer.get_scale()
            np.savetxt(('./quantizer_output/b_scale_out/%f.txt' % time.time()), bias_scale, delimiter='\n')
            #######################输出当前层的量化偏置
            q_bias_txt = self.bias_quantizer.get_quantize_value(bias)
            q_bias_txt = np.array(q_bias_txt.cpu()).reshape(1, -1)
            np.savetxt(('./quantizer_output/q_bias_out/%f.txt' % time.time()), q_bias_txt, delimiter='\n')

            #############偏置重排序
            if self.reorder == True:
                b_para = np.zeros(2048, dtype=int)
                b_para[0:q_bias_txt.size] = q_bias_txt
                # print(b_para.shape)
                # b_para = np.array(b_para.cpu()).reshape(1, -1)
                np.savetxt(('./quantizer_output/q_bias_reorder/%f.txt' % time.time()), b_para, delimiter='\n')
            ################偏置重排序结束

        # 量化卷积
        output = F.conv2d(
            input=input,
            weight=q_weight,
            bias=q_bias,  # 注意，这里加bias，做完整的conv+bn
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
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
            # return output
            pass
        else:
            print(self.activate + "%s is not supported !")

        if self.quantizer_output == True:

            if not os.path.isdir('./quantizer_output/q_activation_out'):
                os.makedirs('./quantizer_output/q_activation_out')
            if not os.path.isdir('./quantizer_output/a_scale_out'):
                os.makedirs('./quantizer_output/a_scale_out')
            if not os.path.isdir('./quantizer_output/q_activation_max'):
                os.makedirs('./quantizer_output/q_activation_max')
            if not os.path.isdir('./quantizer_output/max_activation_count'):
                os.makedirs('./quantizer_output/max_activation_count')
            if not os.path.isdir('./quantizer_output/q_activation_reorder'):
                os.makedirs('./quantizer_output/q_activation_reorder')
            ##################输出当前激活的量化因子
            activation_scale = self.activation_quantizer.get_scale()
            np.savetxt(('./quantizer_output/a_scale_out/%f.txt' % time.time()), activation_scale, delimiter='\n')
            ##################输出当前层的量化激活
            q_activation_txt = self.activation_quantizer.get_quantize_value(output)

            a_para = q_activation_txt
            #############输入特征图重排序
            if self.reorder == True:
                # 重排序参数
                #print("use activation reorder!")
                shape_input = a_para.shape[1]
                num_TN = int(shape_input / self.TN)
                remainder_TN = shape_input % self.TN
                first = True
                reorder_a_para = None
                if self.activate == 'linear':
                    print('layer-linear reorder!')
                    temp = a_para[:, 0:remainder_TN, :, :]
                    temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                    temp = temp.permute(2, 1, 0).contiguous().view(-1)
                    if first:
                        reorder_a_para = temp.clone().cpu().data.numpy()
                        first = False
                    else:
                        reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())
                else:
                    for k in range(num_TN):
                        temp = a_para[:, k * self.TN:(k + 1) * self.TN, :, :]
                        temp = temp.view(temp.shape[1], temp.shape[2], temp.shape[3])
                        temp = temp.permute(2, 1, 0).contiguous().view(-1)
                        if first:
                            reorder_a_para = temp.clone().cpu().data.numpy()
                            first = False
                        else:
                            reorder_a_para = np.append(reorder_a_para, temp.cpu().data.numpy())

                a_para_flatten = reorder_a_para
                #####验证重排序结果的正确性
                '''if a_para_flatten.size == a_para.shape[0] * a_para.shape[1] * a_para.shape[2] * a_para.shape[3]:
                    print("activation convert correctly!")
                else:
                    print("activation convert mismatchingly!")'''

                q_activation_reorder = a_para_flatten
                q_activation_reorder = np.array(q_activation_reorder).reshape(1, -1)
                np.savetxt(('./quantizer_output/q_activation_reorder/%f.txt' % time.time()),
                           q_activation_reorder, delimiter='\n')
            ##########特征图重排序结束

            q_activation_txt = np.array(q_activation_txt.cpu()).reshape(1, -1)
            q_activation_max = [np.max(q_activation_txt)]  # 统计该层的最大值(即查看是否有溢出)
            max_activation_count = [np.sum(abs(q_activation_txt) >= 127)]  # 统计该层溢出的数目
            # q_weight_max = np.argmax(q_weight_txt)
            np.savetxt(('./quantizer_output/max_activation_count/%f.txt' % time.time()),
                       max_activation_count)
            np.savetxt(('./quantizer_output/q_activation_max/%f.txt' % time.time()), q_activation_max)
            np.savetxt(('./quantizer_output/q_activation_out/%f.txt' % time.time()), q_activation_txt,
                       delimiter='\n')

        output = self.activation_quantizer(output)
        return output

    def BN_fuse(self):
        if self.bn:
            # BN融合
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                        self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(
                    self.beta - self.running_mean * self.gamma / torch.sqrt(
                        self.running_var + self.eps))  # b融running
            weight = self.weight * reshape_to_weight(
                self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
        else:
            bias = self.bias
            weight = self.weight
        return weight, bias
