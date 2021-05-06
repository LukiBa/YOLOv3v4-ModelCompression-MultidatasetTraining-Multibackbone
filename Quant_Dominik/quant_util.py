import numpy as np
import torch
import torch.nn as nn
import math
def findShiftScale(val,num_bits):
    # val = x * 2^e
    # e must be a negative integer
    # x must be a positive integer
    if val == math.pow(2, math.ceil(math.log(val)/math.log(2))):
        return math.ceil(math.log(val)/math.log(2)), 1

    e = np.ceil(np.log2(val))
    x = 1

    e_lifo = []
    x_lifo = []

    approx = x * 2**e
    delta = val-approx
    oldloss = np.square(val-approx)

    while True:
        approx = x * 2**e
        delta = val-approx
        loss = np.square(val-approx)

        if loss < oldloss and delta > 0:
            e_lifo.append(e)
            x_lifo.append(x)

        oldloss = loss

        if delta < 0: # Make approximation smaller
            e -= 1
            x *= 2
            x -= 1

        else:
            x += 1

        if x > 2**num_bits or e < -40:
            if len(e_lifo) != 0:
                return e_lifo[-1],x_lifo[-1]
            else: 
                return math.ceil(math.log(val)/math.log(2)), 1
    return 0,0

def fuse(conv, bn):
    import torch

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

#Reject outliers
def reject_outliers(data, m=2):
    mean = torch.mean(data)
    std = torch.std(data)
    for idx, value in np.ndenumerate(data):
        if (abs(value - mean)) > m * std:
            if value - mean > 0:
                data[idx] = m * std
            else:
                data[idx] = -m * std
    return data

#2/3 Max
def squeeze_net(data):
    maxabs = torch.max(torch.abs(data))
    for idx, value in np.ndenumerate(data):
        if abs(data[idx]) > 2/3*maxabs:
            if data[idx] > 0:
                data[idx] = 2/3*maxabs
            else:
                data[idx] = -2/3*maxabs
    return data

#Mean
def mean_scale(data):
    meanabs =  (torch.mean(torch.abs(data)))
    for idx, value in np.ndenumerate(data):
        if abs(data[idx]) > meanabs:
            if data[idx] > 0:
                data[idx] = meanabs
            else:
                data[idx] = -meanabs
    return data

#Power Two
def power_two(data,S,num_bits):
    import math

    max_w = torch.max(torch.abs(data))
    qmax = ((2**num_bits)/2)-1
    qmin = -(qmax+1)
    S["s_w"] = qmax/max_w   

    x = S["s_y"]/(S["s_w"]*S["s_x"])
    next = math.pow(2, math.ceil(math.log(x)/math.log(2)))
    S["s_w"] = S["s_y"] / (S["s_x"] * next)
    max_w = qmax * S["s_w"] 
    # for idx, value in np.ndenumerate(data):
    #     if (abs(value) > max_w):
    #         if value > 0:
    #             data[idx] = max_w
    #         else:
    #             data[idx] = -max_w
    return S["s_w"]




