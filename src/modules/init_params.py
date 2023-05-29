from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from src.modules.multihead_attention import MultiheadAttention
from src.modules.transformer_m_layers import AtomFeature, MoleculeAttnBias, Molecule3DBias, AtomTaskHead
from src.modules.transformer_m_encoder_layer import TransformerMEncoderLayer


def init_params(module):

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
