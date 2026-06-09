from collections import OrderedDict
from typing import Literal

import math
import torch
import torch.nn as nn

from transformers import WavLMModel, WavLMConfig

from . import layer as c_layers



class AbsEncoder(nn.Module):
    def __init__(
        self, with_head: bool = True, num_classes: int = -1, n_hid: int = -1
    ) -> None:
        super().__init__()
        if with_head:
            assert num_classes > 0, f"Vocab size should be > 0, instead {num_classes}"
            assert (
                n_hid > 0
            ), f"Hidden size should be > 0, instead {n_hid}"
            self.classifier = nn.Linear(n_hid, num_classes)
        else:
            self.classifier = nn.Identity()

    def impl_forward(self, *args, **kwargs):
        """Implement the forward funcion w/o classifier"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        out = self.impl_forward(*args, **kwargs)
        if isinstance(out, tuple):
            _co = self.classifier(out[0])
            return (_co,) + out[1:]
        else:
            return self.classifier(out)

class ConformerNet(AbsEncoder):
    """The conformer model with convolution subsampling

    Args:
        num_cells (int): number of conformer blocks
        idim (int): dimension of input features
        hdim (int): hidden size in conformer blocks
        num_classes (int): number of output classes
        conv_multiplier (int): the multiplier to conv subsampling module
        dropout_in (float): the dropout rate to input of conformer blocks (after the linear and subsampling layers)
        res_factor (float): the weighted-factor of residual-connected shortcut in feed-forward module
        d_head (int): dimension of heads in multi-head attention module
        num_heads (int): number of heads in multi-head attention module
        kernel_size (int): kernel size in convolution module
        multiplier (int): multiplier of depth conv in convolution module 
        dropout (float): dropout rate to all conformer internal modules
        delta_feats (bool): True if the input features contains delta and delta-delta features; False if not.
    """

    def __init__(
            self,
            num_cells: int,
            idim: int,
            hdim: int,
            num_classes: int = -1,
            conv: Literal['conv2d', 'conv2d4', 'vgg2l', 'k2_vgg2l', 'funasr_vgg2l', 'none'] = 'conv2d',
            conv_multiplier: int = None,
            dropout_in: float = 0.2,
            res_factor: float = 0.5,
            d_head: int = -1,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1,
            dropout_attn: float = 0.0,
            delta_feats: bool = False,
            causal: bool = False,
            modernized_atten: bool = False,
            with_head: bool = True,
            subsample_norm: str = 'none',
            time_reduction_factor: int = 1,
            time_reduction_pos: int = -1):
        super().__init__(with_head=with_head, num_classes=num_classes, n_hid=hdim)
        assert isinstance(time_reduction_factor, int)
        assert time_reduction_factor >= 1

        if delta_feats:
            in_channel = 3
        else:
            in_channel = 1

        if conv == 'vgg2l':
            self.conv_subsampling = c_layers.VGG2LSubsampling(in_channel)
            ch_sub = math.ceil(math.ceil((idim//in_channel)/2)/2)
            conv_dim = 128 * ch_sub
        elif conv == 'k2_vgg2l':
            self.conv_subsampling = c_layers.K2_VggSubsampling(in_channel)
            conv_dim = [32, 64][-1] * (((idim - 1) // 2 - 1) // 2)
        elif conv == 'funasr_vgg2l':
            self.conv_subsampling = c_layers.Paraformer_Vgg2lSubsampling(in_channel)
            conv_dim = 128 * ((idim // 2) // 2)
        elif conv == 'conv2d':
            if conv_multiplier is None:
                conv_multiplier = hdim
            self.conv_subsampling = c_layers.Conv2dSubdampling(
                conv_multiplier, norm=subsample_norm, stacksup=delta_feats)
            conv_dim = conv_multiplier * (((idim//in_channel)//2)//2)
        elif conv == 'conv2d4':
            if conv_multiplier is None:
                conv_multiplier = hdim
            self.conv_subsampling = c_layers.Conv2dSubdampling4(
                conv_multiplier)
            conv_dim = conv_multiplier * (((idim//in_channel - 1)//2 - 1)  //2)
        elif conv == 'none':
            self.conv_subsampling = None
            conv_dim = idim
        else:
            raise RuntimeError(f"Unknown type of convolutional layer: {conv}")

        self.linear_drop = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(conv_dim, hdim)),
            ('dropout', nn.Dropout(dropout_in))
        ]))

        self.cells = nn.ModuleList()
        if modernized_atten:
            pe = c_layers.LegacyRelPositionalEncoding(d_model=hdim, dropout_rate=dropout)
            #pe = None
        else:
            pe = c_layers.PositionalEncoding(hdim)

        for i in range(num_cells):
            if i == time_reduction_pos and time_reduction_factor > 1:
                cell = c_layers.TimeReduction(time_reduction_factor)
                self.cells.append(cell)

            cell = c_layers.ConformerCell(
                hdim, pe, res_factor, d_head, num_heads, kernel_size, multiplier, dropout, dropout_attn, causal = causal, modernized_atten = modernized_atten)
            self.cells.append(cell)

        if time_reduction_factor > 1 and time_reduction_pos == -1:
            self.cells.append(c_layers.TimeReduction(time_reduction_factor))

    def impl_forward(self, x: torch.Tensor, lens: torch.Tensor):
        if self.conv_subsampling is None:
            x_subsampled, ls = x, lens
        else:
            x_subsampled, ls = self.conv_subsampling(x, lens)
        out = self.linear_drop(x_subsampled)
        
        for cell in self.cells:
            out, ls = cell(out, ls)
        
        return out, ls


class WavLMEncoder(AbsEncoder):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int = -1,
        hdim : int = 768,
        tune_wavlm: bool = True,
        with_head: bool = True,
        freeze_conv: bool = True,
        freeze_encoder: bool = True,
    ):

        super().__init__(
            with_head=with_head,
            num_classes=num_classes,
            n_hid=hdim,
        )
        config = WavLMConfig.from_pretrained(pretrained_model)
        config.mask_time_prob = 0.005
        config.mask_feature_prob = 0.004
        self.wavlm_model = WavLMModel.from_pretrained(pretrained_model, config=config)

        if freeze_conv:
            for p in self.wavlm_model.feature_extractor.parameters():
                p.requires_grad = False

        if freeze_encoder:
            for p in self.wavlm_model.encoder.parameters():
                p.requires_grad = False

        if not tune_wavlm:
            for p in self.wavlm_model.parameters():
                p.requires_grad = False
        
    def impl_forward(self, x: torch.Tensor, xlens: torch.Tensor):
        """
        Args:
            x: (B, T, 1) or (B, T)
            xlens: (B,)
        Returns:
            feats: (B, T', D)
            out_lens: (B,)
        """

        if x.dim() == 3:
            x = x.squeeze(-1)

        outputs = self.wavlm_model(
            x,
            return_dict=True,
        )

        feats = outputs.last_hidden_state  # (B, T', D)
        out_lens = self.wavlm_model._get_feat_extract_output_lengths(xlens)

        return feats, out_lens
