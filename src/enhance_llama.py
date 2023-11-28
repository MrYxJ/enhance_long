# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Github       : https://github.com/MrYxJ
 Date         : 2023-11-27 02:02:08
 LastEditTime : 2023-11-27 10:37:53
 Copyright (C) 2023 mryxj. All rights reserved.
'''

import transformers

from .replace_llama_attention import LlamaAttentionEnhanceLong
from .replace_llama_attention import LlamaFlashAttention2EnhanceLong

def enhance_llama2():
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionEnhanceLong
    print("Replace LlamaAttention with LlamaAttentionEnhanceLong, \
    which default use Consistent Dynamic NTK RoPE and LogN Attention to enhance long context.")
    
    transformers.models.llama.modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2EnhanceLong
    print("Replace LlamaFlashAttention2 with LlamaFlashAttention2EnhanceLong, \
    which default use Consistent Dynamic NTK RoPE and LogN Attention to enhance long context.")
