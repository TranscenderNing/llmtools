# Copyright (c) 2024 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pkg_resources import packaging
from importlib.metadata import version
try:
    import mup
except ImportError:
    pass

# from megatron import mpu
# from megatron.model.fused_softmax import FusedScaleMaskSoftmax
# from megatron.model.activations import get_activation
# from megatron.model.utils import exists, get_fusion_type
# from megatron.model.positional_embeddings import (
#     RotaryEmbedding,
#     apply_rotary_pos_emb_torch,
#     apply_rotary_pos_emb,
#     AliBi,
# )
# from megatron.model.fused_rope import (
#     FusedRoPEFunc,
#     fused_apply_rotary_pos_emb_cached,
# )
# from megatron.model.fused_bias_dropout import (
#     get_bias_dropout_add,
#     bias_dropout_add_fused_train,
#     bias_dropout_add_fused_inference,
# )
# from megatron.model.utils import configure_sparse_attention

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

class Pattention(nn.Module):
    """Pattention Layer.
    """

    def __init__(
        self,
        neox_args,
        input_channels,
        output_channels,
        param_token_num,
        param_key_init_method,
        param_value_init_method,
    ):
        super().__init__()

        self.param_token_num = param_token_num
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = neox_args.norm_activation_type
        
        self.key_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_key_dim), dtype = torch.bfloat16))
        self.value_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_value_dim), dtype = torch.bfloat16))
        
        param_key_init_method(self.key_param_tokens)
        param_value_init_method(self.value_param_tokens)
    
    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax': 
            # NOTE: softmax = exp_l1_norm
            # outputs = F.softmax(inputs, dim=dim) * inputs.shape[dim]
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
            outputs = norm_outputs
        elif normalize_type == 'gelu_l2_norm':
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
            outputs = norm_outputs
        elif normalize_type == 'l2_norm_gelu':
            norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
            nonlinear_outputs = F.gelu(norm_outputs)
            outputs = nonlinear_outputs
        else:
            raise NotImplementedError
        return outputs

    def forward(self, inputs, dropout_p=0.0, router_index=None, attn_mask=None, scale=None):

        query = inputs
        if router_index is None:
            # not MoE mode
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]
        
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale 
        # just for gelu nonlinear, set torch.zeros for softmax
        attn_bias = torch.ones(L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # just for gelu nonlinear, set -inf for softmax
                attn_bias.masked_fill_(attn_mask.logical_not(), 0)
            else:
                raise NotImplementedError

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # just for gelu nonlinear, set attn_weight += attn_bias for softmax
        attn_weight *= attn_bias
        # modified softmax
        attn_weight = self.nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value

        return output
    
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 模拟一个简单的输入，假设 neox_args 和初始化方法为以下内容
class SimpleArgs:
    def __init__(self):
        self.norm_activation_type = 'gelu_l2_norm'  # 或 'softmax', 'gelu_l2_norm', 'l2_norm_gelu'

# 初始化函数
def key_init(param):
    # torch.nn.init.normal_(param, mean=0, std=0.1)
    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    # # 初始化为1.0
    # nn.init.ones_(param)

def value_init(param):
    # torch.nn.init.normal_(param, mean=0, std=0.1)
    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    # # 初始化为1.0
    # nn.init.ones_(param)
    
    
    
def init_method_normal(sigma, use_mup_outer=False, mup_init_scale=1.0):
    """Init method based on N(0, sigma)."""

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.normal_(tensor, mean=0.0, std=sigma)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_
                    




def get_test_pattention():
    neox_args = SimpleArgs()
    input_channels = 4096  # 输入通道数
    output_channels = 4096   # 输出通道数
    param_token_num = 8  # 令牌数量
    
    # 创建 Pattention 层
    patten = Pattention(
        neox_args=neox_args,
        input_channels=input_channels,
        output_channels=output_channels,
        param_token_num=param_token_num,
        param_key_init_method=key_init,
        param_value_init_method=value_init
    )
    return patten
    
# 测试 Pattention 类
def test_pattention():
    # 模拟一些必要的参数
    neox_args = SimpleArgs()
    input_channels = 64  # 输入通道数
    output_channels = 64   # 输出通道数
    param_token_num = 8  # 令牌数量
    dropout_p = 0.1  # dropout 概率
    
    # 创建 Pattention 层
    patten = Pattention(
        neox_args=neox_args,
        input_channels=input_channels,
        output_channels=output_channels,
        param_token_num=param_token_num,
        param_key_init_method=key_init,
        param_value_init_method=value_init
    )
    
    for name, param in patten.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} shape: {param.shape}")
    
    # 创建输入张量 (L, C) -> (batch_size, seq_len, channels)
    batch_size = 2
    seq_len = 5
    inputs = torch.randn(batch_size, seq_len, input_channels)  # 随机初始化输入

    # 前向传播
    output = patten(inputs, dropout_p=dropout_p)


    # 输出期望值：我们期望的输出形状是 (batch_size, seq_len, output_channels)
    assert output.shape == (batch_size, seq_len, output_channels), f"Expected shape (2, 5, 64), but got {output.shape}"

    return output

# 运行测试
# output = test_pattention()
# print(output.shape)



