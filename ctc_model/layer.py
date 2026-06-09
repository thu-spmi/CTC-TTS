# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Hongyu Xiang,
#         Keyu An,
#         Huahuan Zheng (maxwellzh@outlook.com)

"""basic nn layers impl"""

import math
import pickle
from collections import OrderedDict
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import numpy


class StackDelta(nn.Module):
    """Stack the features from 120 into 40 x 3

    in: [batch, len, 120]
    out: [batch, 3, len, 40]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3
        assert x.size(2) % 3 == 0

        x = x.view(x.size(0), x.size(1), 3, x.size(2)//3)
        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()
        return out


class UnStackDelta(nn.Module):
    """Reverse of StackDelta"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4

        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), out.size(2) * out.size(3))
        return out


'''
Conv2dSubsampling: From Google TensorflowASR
https://github.com/TensorSpeech/TensorFlowASR/blob/5fbd6a89b93b703888662f5c47d05bae256e98b0/tensorflow_asr/models/layers/subsampling.py
Originally wrote with Tensorflow, translated into PyTorch by Huahuan.
'''


class Unsqueeze(nn.Module):
    """For pickle model"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class Conv2dSubdampling(nn.Module):
    def __init__(self, multiplier: int, norm: Literal['batch', 'layer', 'none'] = 'none', stacksup: bool = False):
        super().__init__()
        self._lens_in_args_ = None

        if stacksup:
            self.stack = StackDelta()
            idim = 3
        else:
            self.stack = Unsqueeze(1)
            idim = 1

        if norm == 'none':
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
                nn.Conv2d(idim, multiplier, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
                nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2),
                nn.ReLU()
            )
            return
        elif norm == 'batch':
            Norm = nn.BatchNorm2d
        elif norm == 'layer':
            Norm = nn.LayerNorm
        else:
            raise RuntimeError(
                f"Unknown normalization method: {norm}, expect one of ['batch', 'layer', 'none']")
        self.conv = nn.Sequential(
            nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
            nn.Conv2d(idim, multiplier, kernel_size=3, stride=2),
            Norm(multiplier),
            nn.ReLU(),
            nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0.),
            nn.Conv2d(multiplier, multiplier, kernel_size=3, stride=2),
            Norm(multiplier),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4 (not strictly).
        """
        # [B, T, D] -> [B, C, T, D]
        x = self.stack(x)
        # [B, C, T, D] -> [B, OD, T//4, D//4]
        # start_time = time.time()
        out = self.conv(x)
        # end_time = time.time()
        # print("conv2d", end_time - start_time)
        B, OD, NT, ND = out.size()
        # [B, OD, T//4, D//4] -> [B, T//4, OD, D//4]
        out = out.permute(0, 2, 1, 3)
        # [B, T//4, OD, D//4] -> [B, T//4, OD * D//4]
        out = out.contiguous().view(B, NT, OD*ND)
        lens_out = torch.div(lens, 2, rounding_mode='floor')
        lens_out = torch.div(lens_out, 2, rounding_mode='floor').to(lens.dtype)
        return out, lens_out


'''NOTE (Ziwei):
Conv2dSubsampling4: From Wenet ASR.
Reference:
https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/subsampling.py 
In Wenet impl, there is no nn.ConstantPad2d.
'''


class Conv2dSubdampling4(nn.Module):
    def __init__(self, multiplier: int):
        super(Conv2dSubdampling4, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=multiplier, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=multiplier, out_channels=multiplier, kernel_size=3, stride=2),
                nn.ReLU()
            )

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """
        Convolutional 2D subsampling (to 1/4 length)
        """
        # [B, C, T, D] -> [B, OD, T//4, D//4]
        #start_time = time.time()
        out = self.conv(x.unsqueeze(1))
        #end_time = time.time()
        #print(f'1/4 conv2d subsampling times:{end_time - start_time:.6f} seconds')
        batch_size, channels, subsampled_lengths, sumsampled_dim = out.size()
        # [B, OD, T//4, D//4] -> [B, T//4, OD, D//4]
        out = out.transpose(1,2)
        # [B, T//4, OD, D//4] -> [B, T//4, OD * D//4]
        out = out.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        
        lens_out = lens >> 2
        lens_out -= 1
        
        return out, lens_out
    
    

'''NOTE (Huahuan):
VGG2L with 1/4 subsample of time dimension.
In ESPNET impl, there is no normalization, so I just follow it.
Reference:
https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/rnn/encoders.html#VGG2L
'''


class VGG2LSubsampling(nn.Module):
    def __init__(self, in_channel: int = 1):
        '''VGG 1/4 subsample layer

        Input: shape (B, T, D)
        Ouput: shape (B, T//4, 128*(D//C//4)),
            C: in_channel, ensure `D % C == 0`

        '''
        super().__init__()

        self.vgg_conv = nn.Sequential(
            # first block
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            # second block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.in_channel = in_channel

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # [B, T, D] -> [B, C, T, D//C]
        x = x.view(
            x.size(0),
            x.size(1),
            self.in_channel,
            x.size(2) // self.in_channel,
        ).transpose(1, 2)

        # [B, C, T, D//C] -> [B, 128, T//4, D//C//4]
        conv_out = self.vgg_conv(x)  # type:torch.Tensor
        # [B, 128, T//4, D//C//4] -> [B, T//4, 128, D//C//4]
        transposed_out = conv_out.transpose(1, 2)
        # [B, T//4, 128, D//C//4] -> [B, T//4, 128 * (D//C//4)]
        contiguous_out = transposed_out.contiguous().view(
            transposed_out.size(0), transposed_out.size(1), -1)

        lens_out = torch.ceil(torch.ceil(lens/2).to(lens.dtype)/2)
        return contiguous_out, lens_out


'''NOTE (Ziwei):
VGG2L with 1/4 subsample of time dimension.
Reference:
https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/conformer_ctc/subsampling.py
'''


class K2_VggSubsampling(nn.Module):
    """Trying to follow the setup described in the following paper:
    https://arxiv.org/pdf/1910.09799.pdf

    This paper is not 100% explicit so I am guessing to some extent,
    and trying to compare with other VGG implementations.

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' = T//4
    """
    def __init__(self, in_channel: int = 1):
        super(K2_VggSubsampling, self).__init__()

        layers = []
        block_dims = [32, 64]

        for block_dim in block_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Conv2d(
                    in_channels=block_dim,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                )
            )
            layers.append(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )
            in_channel = block_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        x = x.unsqueeze(1)
        x = self.layers(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        lens_out = (lens - 1) // 2 - 1
        lens_out = (lens_out - 1) // 2 - 1
        lens_out = torch.clamp(lens_out, min=1).to(lens.dtype)  # Ensure no non-positive lengths

        return x, lens_out


'''NOTE (Ziwei):
VGG2L with 1/4 subsample of time dimension.
Reference:
https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/utils/vgg2l.py
'''


class Paraformer_Vgg2lSubsampling(nn.Module):
    """VGG2LSubsampling 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, in_channel: int = 1):
        super(Paraformer_Vgg2lSubsampling, self).__init__()
        self.vgg_conv = nn.Sequential(
            # first block
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 2)),
            # second block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, x: torch.Tensor, lens: torch.Tensor):

    
        x = x.unsqueeze(1)
    
        x = self.vgg_conv(x)
        
        b, c, t, f = x.size()
        
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
    
        lens_out = torch.div(lens, 3, rounding_mode='floor')  # First pool layer
        lens_out = torch.div(lens_out, 2, rounding_mode='floor')  # Second pool layer
        lens_out = lens_out.to(lens.dtype)
        

        return x, lens_out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4096):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:x.size(1), :]


'''
Relative positional multi-head attention implementation of Transformer-XL
from huggingface
https://github.com/huggingface/transformers/blob/1bdf42409c452a767ac8e2119bceb8f5c704c8f1/src/transformers/models/transfo_xl/modeling_transfo_xl.py
And part of the codes are modified according to TensorflowASR impl
https://github.com/TensorSpeech/TensorFlowASR/blob/5fbd6a89b93b703888662f5c47d05bae256e98b0/tensorflow_asr/models/layers/multihead_attention.py
'''


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(
        self,
        idim: int,
        n_head: int,
        d_head: int,
        pe: PositionalEncoding,
        dropatt: float = 0.0
    ):

        super().__init__()

        self.pe = pe

        self.n_head = n_head
        self.d_model = idim
        self.d_head = d_head

        self.call_qkv = nn.Linear(idim, 3 * n_head * d_head, bias=False)

        self.dropoutatt = nn.Dropout(dropatt)
        self.linearout = nn.Linear(n_head * d_head, idim, bias=False)

        self.scale = 1 / (d_head ** 0.5)

        self.r_r_bias = nn.Parameter(
            torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(
            torch.FloatTensor(self.n_head, self.d_head))

        torch.nn.init.xavier_uniform_(self.r_r_bias)
        torch.nn.init.xavier_uniform_(self.r_w_bias)

        self.linearpos = nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    @staticmethod
    def _rel_shift(x: torch.Tensor):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = x.new_zeros(zero_pad_shape)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, word_embedding: torch.Tensor, lens: torch.Tensor, mems=None):
        """Relative positional embedding multihead attention

        B: batch size
        T: (maximum) sequence length
        E: word embedding size (also the postional embedding size)
        C: memory context length, K=C+T

        Args:
            word_embedding (torch.Tensor): input segment (word) embedding, size [B, T, E]
            lens (torch.Tensor): input sequence lengths, used to generate attention mask, size [B]
            mems (torch.Tensor, None): if not None, memory tensors, size [B, C, E]
        
        Returns:
            torch.Tensor: attention out

        """
        '''
        word_embedding: [B, T, E] -> [T, B, E]
        mems: [B, C, E] -> [C, B, E] (if not None)
        '''
        word_embedding = word_embedding.transpose(0, 1).contiguous()
        len_seq, batchsize = word_embedding.size()[:2]
        len_k = len_seq
        if all(le == lens[0] for le in lens):
            # set mask to None
            attn_mask = None
            if mems is not None:
                mems = mems.transpose(0, 1).contiguous()
                len_k += mems.size(0)
        else:
            if mems is not None:
                mems = mems.transpose(0, 1).contiguous()
                lens = lens + mems.size(0)
                len_k += mems.size(0)
            # attn_mask: [K, B]_{1,0}
            attn_mask = torch.arange(len_k, device=word_embedding.device)[
                :, None] >= lens[None, :].to(word_embedding.device)

        # pos_embedding: [K, E]
        pos_enc = self.pe(torch.empty((1, len_k)))

        if mems is not None:
            # embed_with_mem: cat([C, B, E], [T, B, E]) -> [C+T, B, E] = [K, B, E]
            embed_with_mem = torch.cat([mems, word_embedding], dim=0)
            '''
            W_heads: f([K, B, E]) -> [K, B, 3*H*D]
                H: n_heads
                D: d_heads
            '''
            W_heads = self.call_qkv(embed_with_mem)

            # R_head_k: f([K, E]) -> [K, H*D]
            R_head_k = self.linearpos(pos_enc)

            # W_head_q/W_head_k/W_head_v: f([K, B, 3HD]) -> [K, B, HD]
            W_head_q, W_head_k, W_head_v = torch.chunk(W_heads, 3, dim=-1)

            # W_head_q: f([K, B, HD]) -> [T, B, HD]
            W_head_q = W_head_q[-len_seq:]

        else:
            # W_heads: f([T, B, E]) -> [T, B, 3*H*D]
            W_heads = self.call_qkv(word_embedding)

            # R_head_k: f([T, E]) -> [T, H*D]
            R_head_k = self.linearpos(pos_enc)

            # W_head_q/W_head_k/W_head_v: f([T, B, 3HD]) -> [T, B, HD]
            W_head_q, W_head_k, W_head_v = torch.chunk(W_heads, 3, dim=-1)

        # W_head_q: [T, B, HD] -> [T, B, H, D]
        W_head_q = W_head_q.view(len_seq, batchsize, self.n_head, self.d_head)

        # W_head_k/W_head_v: [K, B, HD] -> [K, B, H, D]
        W_head_k = W_head_k.view(len_k, batchsize, self.n_head, self.d_head)
        W_head_v = W_head_v.view_as(W_head_k)

        # R_head_k: [K, H*D] -> [K, H, D]
        R_head_k = R_head_k.view(len_k, self.n_head, self.d_head)

        # compute attention score

        # RW_head_q/RR_head_q: [T, B, H, D]
        RW_head_q = W_head_q + self.r_w_bias
        RR_head_q = W_head_q + self.r_r_bias

        # FIXME: torch.einsum is not optimized, which might cause slow computation
        # AC: f([T, B, H, D], [K, B, H, D]) -> [T, K, B, H]
        AC = torch.einsum("ibnd,jbnd->ijbn", (RW_head_q, W_head_k))

        # BD: f([T, B, H, D], [K, H, D]) -> [T, K, B, H]
        BD = torch.einsum("ibnd,jnd->ijbn", (RR_head_q, R_head_k))

        # BD: [T, K, B, H] -> [T, K, B, H]
        BD = self._rel_shift(BD)

        # attn_score: [T, K, B, H]
        attn_score = AC + BD
        attn_score *= self.scale

        # compute attention probability
        if attn_mask is not None:
            # use in-plcae fill
            attn_score = attn_score.masked_fill_(
                attn_mask[None, :, :, None], -float('inf'))

        # attn_prob: f([T, K, B, H]) -> [T, K, B, H]
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.dropoutatt(attn_prob)

        # compute attention vector

        # attn_vec: f([T, K, B, H], [K, B, H, D]) -> [T, B, H, D]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, W_head_v))

        # attn_vec: [T, B, H, D] -> [T, B, HD]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1),
            self.n_head * self.d_head)

        # attn_out: f([T, B, HD]) -> [T, B, E]
        attn_out = self.linearout(attn_vec)

        # attn_out: [T, B, E] -> [B, T, E]
        attn_out = attn_out.transpose(0, 1).contiguous()

        return attn_out


class StandardRelPositionalMultiHeadAttention(RelPositionMultiHeadAttention):
    def __init__(self, idim: int, n_head: int, pe: PositionalEncoding, dropatt: float = 0):
        super().__init__(idim, n_head, idim//n_head, pe, dropatt=dropatt)


class FFModule(nn.Module):
    """Feed-forward module

    default output dimension = idim
    x0 -> LayerNorm -> FC -> Swish -> Dropout -> FC -> Dropout -> x1
    x0 + res_factor * x1 -> output
    """

    def __init__(self, idim: int, res_factor: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        assert res_factor > 0. and dropout >= 0.
        self._res_factor = res_factor

        self.ln = nn.LayerNorm([idim])
        self.fc0 = nn.Linear(idim, idim*4)
        self.swish = nn.SiLU()
        self.dropout0 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(idim*4, idim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        output = self.ln(x)
        output = self.fc0(output)
        output = self.swish(output)
        output = self.dropout0(output)
        output = self.fc1(output)
        output = self.dropout1(output)
        output = x + self._res_factor * output

        return output


class ConvModule(nn.Module):
    def __init__(self, idim: int, kernel_size: int = 32, dropout: float = 0.0, multiplier: int = 1, causal: bool = False) -> None:
        super().__init__()

        self.ln = nn.LayerNorm([idim])
        self.pointwise_conv0 = nn.Conv1d(
            idim, 2 * idim, kernel_size=1, stride=1)
        self.glu = nn.GLU(dim=1)
        cdim = idim
        if causal:
            padding = 0
        else:
            padding = (kernel_size-1)//2
        self.padding = nn.ConstantPad2d(
            (padding, kernel_size-1-padding, 0, 0), 0.)
        self.depthwise_conv = nn.Conv1d(
            cdim, multiplier*cdim, kernel_size=kernel_size, stride=1, groups=cdim, padding=0)
        cdim = multiplier * cdim
        self.bn = nn.BatchNorm1d(cdim)
        self.swish = nn.SiLU()
        self.pointwise_conv1 = nn.Conv1d(cdim, idim, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # [B, T, D]
        output = self.ln(x)
        # [B, T, D] -> [B, D, T]
        output = output.transpose(1, 2)
        # [B, D, T] -> [B, 2*D, T]
        output = self.pointwise_conv0(output)
        # [B, 2D, T] -> [B, D, T]
        output = self.glu(output)
        # [B, D, T] -> [B, multiplier*D, T]
        output = self.padding(output)
        output = self.depthwise_conv(output)
        if output.size(0) > 1 or output.size(2) > 1:
            # Doing batchnorm with [1, D, 1] tensor raise error.
            output = self.bn(output)
        output = self.swish(output)
        # [B, multiplier*D, T] -> [B, D, T]
        output = self.pointwise_conv1(output)
        output = self.dropout(output)
        # [B, D, T] -> [B, T, D]
        output = output.transpose(1, 2)

        return x + output


class MHSAModule(nn.Module):
    def __init__(self, hdim, d_head: int, num_heads: int, pe: PositionalEncoding, dropout: float = 0.0, dropout_attn: float = 0.0):
        super().__init__()

        self.ln = nn.LayerNorm(hdim)
        if d_head == -1:
            # a "standard" multi-head attention
            self.mha = StandardRelPositionalMultiHeadAttention(
                hdim, num_heads, pe, dropout_attn)
        else:
            self.mha = RelPositionMultiHeadAttention(
                hdim, num_heads, d_head, pe, dropout_attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lens: torch.Tensor, mems=None):
        x_norm = self.ln(x)
        attn_out = self.mha(x_norm, lens, mems)
        attn_out = self.dropout(attn_out)
        return x + attn_out, lens


class ConformerCell(nn.Module):
    def __init__(
            self,
            idim: int,
            pe: PositionalEncoding,
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1,
            dropout_attn: float = 0.0,
            causal: bool = False,
            modernized_atten: bool = False):
        super().__init__()

        self.pe = pe
        self.modernized_atten = modernized_atten
        self.ffm0 = FFModule(idim, res_factor, dropout)
        
        if modernized_atten:
            self.mhsam = LegacyRelPositionMultiHeadedAttention(n_head=num_heads, n_feat=idim, dropout_rate=dropout_attn)
            #self.mhsam = MultiHeadedSelfAttentionModule(d_model=idim, num_heads=num_heads, dropout_p=dropout_attn)
        else:
            self.mhsam = MHSAModule(idim, d_head, num_heads,
                                pe, dropout, dropout_attn)
        self.convm = ConvModule(
            idim, kernel_size, dropout, multiplier, causal = causal)
        self.ffm1 = FFModule(idim, res_factor, dropout)
        self.ln = nn.LayerNorm(idim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        
        ffm0_out = self.ffm0(x)
       
        if self.modernized_atten:
            pos_emb = self.pe(ffm0_out)
            ffm0_out_ln = self.ln(ffm0_out)
            attn_out = ffm0_out + self.mhsam(ffm0_out_ln, ffm0_out_ln, ffm0_out_ln, pos_emb, mask = None)

            ## Latest
            #attn_out = self.mhsam(ffm0_out, mask = None)
            attn_ls = lens
        else:
            attn_out, attn_ls = self.mhsam(ffm0_out, lens)
        
        conv_out = self.convm(attn_out)
       
        ffm1_out = self.ffm1(conv_out)
        
        out = self.ln(ffm1_out)
        
        return out, attn_ls


class VGG2L(nn.Module):
    def __init__(self, in_channel=4):
        super(VGG2L, self).__init__()
        kernel_size = 3
        padding = 1
        self.conv1_1 = torch.nn.Conv2d(
            in_channel, 64, kernel_size, stride=1, padding=padding)
        self.conv1_2 = torch.nn.Conv2d(
            64, 64, kernel_size, stride=1, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2_1 = torch.nn.Conv2d(
            64, 128, kernel_size, stride=1, padding=padding)
        self.conv2_2 = torch.nn.Conv2d(
            128, 128, kernel_size, stride=1, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = self.bn1(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1, 2], stride=[1, 2], ceil_mode=True)
        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = self.bn2(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1, 2], stride=[1, 2], ceil_mode=True)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens


class Lookahead(nn.Module):
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x


class TDNN(nn.Module):
    def __init__(self, idim: int, odim: int, half_context: int = 1, dilation: int = 1, stride: int = 1):
        super().__init__()

        self.stride = stride
        if stride > 1:
            dilation = 1
            padding = 0
        else:
            padding = half_context * dilation

        self.conv = torch.nn.Conv1d(
            idim, odim, 2*half_context+1, stride=stride, padding=padding, dilation=dilation)
        # NOTE (Huahuan): I think we should use layernorm instead of batchnorm
        # self.bn = MaskedBatchNorm1d(odim, eps=1e-5, affine=True)
        self.ln = nn.LayerNorm(odim)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tdnn_in = x.transpose(1, 2)
        if self.stride > 1:
            ilens = ilens // self.stride

        tdnn_out = self.conv(tdnn_in)

        output = F.relu(tdnn_out)
        output = output.transpose(1, 2)
        output = self.ln(output)
        return output, ilens


class _LSTM(nn.Module):
    def __init__(self, idim, hdim, n_layers, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(idim, hdim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        self.lstm.flatten_parameters()

        packed_input = pack_padded_sequence(
            x, ilens.to("cpu"), batch_first=True)
        packed_output, _ = self.lstm(packed_input, hidden)
        out, olens = pad_packed_sequence(packed_output, batch_first=True)

        return out, olens


class TimeReduction(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        assert N >= 1 and isinstance(N, int)
        if N == 1:
            print("WARNING: TimeReduction factor was set to 1.")
        self.N = N

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        return x[:, ::self.N, :].clone(), torch.div(x_lens-1, self.N, rounding_mode='floor') + 1


class CausalConv2d(nn.Module):
    """
    Causal 2d-conv. Applied to (N, C, T, U) dim tensors.

    Args:
        in_channels  (int): the input dimension (namely K)
        out_channels (int): the output dimension
        kernel_size  (int): the kernel size (kernel is square)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], islast: bool = False):
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"Invalid initialization for CausalConv2d: {in_channels}, {out_channels}, {kernel_size}")

        if islast:
            self.causal_conv = nn.Sequential(OrderedDict([
                # seperate convlution
                ('depth_conv', nn.Conv2d(in_channels, in_channels,
                 kernel_size=kernel_size, groups=in_channels)),
                ('point_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1))
                # 'conv': nn.Conv2d(in_channels, out_channels, kernel_size)
            ]))
        else:
            # NOTE (Huahuan): I think a normalization is helpful so that the padding won't change the distribution of features.
            self.causal_conv = nn.Sequential(OrderedDict([
                # seperate convlution
                ('depth_conv', nn.Conv2d(in_channels, in_channels,
                 kernel_size=kernel_size, groups=in_channels)),
                ('point_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                ('relu', nn.ReLU(inplace=True)),
                ('bn', nn.BatchNorm2d(in_channels)),
                # 'conv': nn.Conv2d(in_channels, out_channels, kernel_size)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.causal_conv(x)


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.
    
    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
    
    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
        .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's BatchNorm1d implementation for argument details.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(num_features, eps, momentum,
                                                affine, track_running_stats)

    def forward(self, inp, lengths):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp**2).sum([0, 2]) - mean**2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] +
                                                        self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp


class SampledSoftmax(nn.Module):
    """
    Conduct softmax on given target labels (to reduce memory consumption).

    Args:
        blank (int): CTC/RNN-T blank index, default -1.
        uniform_ratio (float): uniformly draw indices apart from target labels via given ratio.
    """

    def __init__(self, blank: int = -1, uniform_ratio: Optional[float] = 0.0) -> None:
        super().__init__()
        assert isinstance(blank, int)

        if blank != -1:
            # CTC / RNN-T keep the first place for <blk>, currently only support <blk>=0
            assert blank == 0
            self.pad = nn.ConstantPad1d((1, 0), 0)

        self.blank_id = blank

        assert isinstance(uniform_ratio, (int, float))
        assert uniform_ratio >= 0 and uniform_ratio < 1
        self.uniform_ratio = float(uniform_ratio)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):

        orin_shape = labels.shape
        if labels.dim() > 1:
            labels = labels.flatten()

        dtype = labels.dtype
        if self.uniform_ratio == 0.0:
            indices_sampled, labels = torch.unique(
                labels, sorted=True, return_inverse=True)
        else:
            n_uniform = int(x.size(-1) * self.uniform_ratio)
            assert n_uniform > 0
            concat_labels = torch.cat(
                [
                    labels,
                    torch.randperm(x.size(-1), device=labels.device,
                                   dtype=dtype)[:n_uniform]
                ],
                dim=0
            )
            indices_sampled, concat_labels = torch.unique(
                concat_labels, sorted=True, return_inverse=True)
            labels = concat_labels[:labels.size(0)]

        indices_sampled = indices_sampled.to(torch.long)
        labels = labels.to(dtype)

        if self.blank_id != -1:
            if indices_sampled[0] != self.blank_id:
                indices_sampled = self.pad(indices_sampled)
                labels += 1

        return x[..., indices_sampled].log_softmax(dim=-1), labels.reshape(orin_shape)


class SyllableEmbedding(nn.Module):
    """The char to syllable mapping is generated by pypinyin module."""

    def __init__(self, num_classes: int, dim_emb: int, syllable_data: str) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(num_classes, dim_emb)

        with open(syllable_data, 'rb') as fib:
            data = pickle.load(fib)
        self.register_buffer('converter', torch.from_numpy(
            data['converter']), persistent=False)
        self.syllable_embedding = nn.Embedding(
            data['num_syllables'], dim_emb)
        del data

    def forward(self, x: torch.Tensor):
        xshape = x.shape
        x_syllable = self.converter[x.view(-1)].reshape(xshape)
        return self.char_embedding(x) + self.syllable_embedding(x_syllable)

class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sequential(inputs)


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * 4, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * 4, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        #start_time = time.time()
        #end_time = time.time()
        #print(f'Feed Forward Module 1 time: {end_time-start_time:.6f} seconds')
        return self.sequential(inputs)
        
    

class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.LN = nn.LayerNorm(in_channels)
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=in_channels * expansion_factor, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=True
                      ),
            nn.GLU(dim=1),
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=in_channels, 
                      kernel_size=kernel_size,
                      groups=in_channels, 
                      stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=True),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=in_channels,
                      kernel_size=1,
                      stride=1, 
                      padding=0, 
                      bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # [B, T, D]
        inputs = self.LN(inputs)
        # [B, T, D] -> [B, D, T]
        inputs = inputs.transpose(1, 2)
        #start_time = time.time()
        #end_time = time.time()
        #print(f'Conformer Convolution Module time:{end_time - start_time:.6f} seconds')
        return self.sequential(inputs).transpose(1, 2)
    

class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()

        # The latest version RepositionalEncoding
        self.positional_encoding = RelPositionalEncoding(d_model)

        # From Funasr LegacyRelPositionalEncoding
        #self.positional_encoding = LegacyRelPositionalEncoding(d_model=d_model, dropout_rate=0.0)

        self.layer_norm = nn.LayerNorm(d_model)

        # The latest version Attention
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        
        # From Funasr Attention
        #self.attention =  LegacyRelPositionMultiHeadedAttention(n_head=num_heads, n_feat=d_model, dropout_rate=dropout_p)
        
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        
        # The latest version RepositioanlEncoding
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        
        # From Funasr LegacyRelPositionalEncoding
        # pos_embedding = self.positional_encoding(inputs)
        # pos_embedding = pos_embedding[1]
        
        inputs_ln = self.layer_norm(inputs)
        outputs = self.attention(inputs_ln, inputs_ln, inputs_ln, pos_embedding, mask=mask)
        

        return inputs + self.dropout(outputs)
    

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
    

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """


    # class Linear(nn.Module):
    #     """
    #     Wrapper class of torch.nn.Linear
    #     Weight initialize by xavier initialization and bias initialize to zeros.
    #     """
    #     def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
    #         super(RelativeMultiHeadAttention.Linear, self).__init__()
    #         self.linear = nn.Linear(in_features, out_features, bias=bias)
    #         torch.nn.init.xavier_uniform_(self.linear.weight)
    #         if bias:
    #             torch.nn.init.zeros_(self.linear.bias)

    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         return self.linear(x)
    

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score
    

'''
copy from Funasr 
'''

class SinusoidalPositionEncoder(torch.nn.Module):
    '''

    '''
    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

    def encode(self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (depth / 2 - 1)
        inv_timescales = torch.exp(torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment))
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps+1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding
    
class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, in_feat, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)
        
        
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(1, 2)  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(1, 2)  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(1, 2)  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        
        return att_outs + fsmn_memory


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        """Initialize MultiSequential with layer_drop.

        Args:
            layer_drop_rate (float): Probability of dropping out each fn (layer).

        """
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        """Repeat."""
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args

class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, mask, mask_shfit_chunk=mask_shfit_chunk, mask_att_chunk_encoder=mask_att_chunk_encoder)), dim=-1)
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(x, mask, mask_shfit_chunk=mask_shfit_chunk, mask_att_chunk_encoder=mask_att_chunk_encoder)
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(x, mask, mask_shfit_chunk=mask_shfit_chunk, mask_att_chunk_encoder=mask_att_chunk_encoder)
                )
        if not self.normalize_before:
            x = self.norm1(x)
        
        residual = x
        
        if self.normalize_before:
            
            x = self.norm2(x)
        
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        
        if not self.normalize_before:
            x = self.norm2(x)
        
        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder
    

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )
    
def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask

def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
        layer_drop_rate (float): Probability of dropping out each fn (layer).

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)

def get_activation(act):
    """Return activation function."""

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()


class Swish(torch.nn.Module):
    """Swish activation definition.

    Swish(x) = (beta * x) * sigmoid(x)
                 where beta = 1 defines standard Swish activation.

    References:
        https://arxiv.org/abs/2108.12943 / https://arxiv.org/abs/1710.05941v1.
        E-swish variant: https://arxiv.org/abs/1801.07145.

    Args:
        beta: Beta parameter for E-Swish.
                (beta >= 1. If beta < 1, use standard Swish).
        use_builtin: Whether to use PyTorch function if available.

    """

    def __init__(self, beta: float = 1.0, use_builtin: bool = False) -> None:
        super(Swish, self).__init__()

        self.beta = beta
        self.use_builtin = use_builtin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        if self.beta > 1:
            return (self.beta * x) * torch.sigmoid(x)
        elif self.use_builtin:
            return torch.nn.functional.silu(x)
        else:
            return x * torch.sigmoid(x)


class Funasr_positionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(Funasr_positionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class LegacyRelPositionalEncoding(Funasr_positionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            reverse=True,
        )

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(pos_emb)
    

class Funasr_Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Funasr_Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else Funasr_positionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
       
        x = self.conv(x)
        
        b, c, t, f = x.size()
        
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Funasr_VGG2LSubsampling(nn.Module):
    """VGG2LSubsampling 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        super(Funasr_VGG2LSubsampling, self).__init__()
        self.vgg_conv = nn.Sequential(
            # first block
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 2)),
            # second block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(128 * ((idim // 2) //2 ), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)
        start_time = time.time()
        x = self.vgg_conv(x)
        end_time = time.time()
        #print(f'1/4 VGG2L_funasr times {end_time - start_time:.3f} seconds')
        b, c, t, f = x.size()
        
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
    
        if x_mask is None:
            return x, None
        else:
            vgg_mask = self.create_new_mask(feats_mask=x_mask)
        return x, vgg_mask
    
    def create_new_mask(self, feats_mask: torch.Tensor) -> torch.Tensor:
        """Create a subsampled mask of feature sequences.

        Args:
            feats_mask: Mask of feature sequences. (B, 1, F)

        Returns:
            vgg_mask: Mask of VGG2L output sequences. (B, 1, sub(F))

        """
        vgg1_t_len = feats_mask.size(2) - (feats_mask.size(2) % 3)
        vgg_mask = feats_mask[:, :, :vgg1_t_len][:, :, ::3]

        vgg2_t_len = vgg_mask.size(2) - (vgg_mask.size(2) % 2)
        vgg_mask = vgg_mask[:, :, :vgg2_t_len][:, :, ::2]

        return vgg_mask

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.h_kv = n_head

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def _forward_linearx(self, name: str, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3
        if name == 'query':
            x = self.linear_q(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h, self.d_k])
        elif name == 'key':
            x = self.linear_k(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        else:
            assert name == 'value'
            x = self.linear_v(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])

        # split last dim
        x = x.view(x_shape)
        x = x.transpose(-3, -2)  # (batch, ...,  head or head_kv, time, d_k)
        return x
    
    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, ..., time1, size).
            key (torch.Tensor): Key tensor (#batch, ..., time2, size).
            value (torch.Tensor): Value tensor (#batch, ..., time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, ..., n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, ..., n_head_kv, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, ..., n_head_kv, time2, d_k).

        """
        q = self._forward_linearx('query', query)
        k = self._forward_linearx('key', key)
        v = self._forward_linearx('value', value)
        return q, k, v
    
    # def forward_qkv(self, query, key, value):
    #     """Transform query, key and value.

    #     Args:
    #         query (torch.Tensor): Query tensor (#batch, time1, size).
    #         key (torch.Tensor): Key tensor (#batch, time2, size).
    #         value (torch.Tensor): Value tensor (#batch, time2, size).

    #     Returns:
    #         torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
    #         torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
    #         torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

    #     """
    #     n_batch = query.size(0)
    #     q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
    #     k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
    #     v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
    #     q = q.transpose(1, 2)  # (batch, head, time1, d_k)
    #     k = k.transpose(1, 2)  # (batch, head, time2, d_k)
    #     v = v.transpose(1, 2)  # (batch, head, time2, d_k)

    #     return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        #n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1).type_as(value)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(-3, -2).contiguous()  # [batch, ..., time1, head, d_k]
        x_shape = x.size()[:-2] + torch.Size([self.h * self.d_k])
        x = x.view(x_shape)  # (batch, ..., time1, d_model)

        #x = (
        #    x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        #)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)
    

class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        
        '''''Funasr demo
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        
        '''''
        
        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class Funasr_ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(Funasr_ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
    

class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
            self,
            size,
            self_attn,
            feed_forward,
            feed_forward_macaron,
            conv_module,
            dropout_rate,
            normalize_before=True,
            concat_after=False,
            stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                ffm_start = time.time()
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            ffm_end = time.time()
            #print(f'Feed Forward Module 1 time:{ffm_end - ffm_start:.6f} seconds')
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        mult_start = time.time()
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)
        mult_end = time.time()
        #print(f'Multi-Headed Self-Attention Module time:{mult_end - mult_start:.6f} seconds')
        # convolution module
        if self.conv_module is not None:
            conv_start = time.time()
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)
            conv_end = time.time()
            #print(f'Conformer Convolution Module time {conv_end - conv_start:.6f} seconds')


        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        feed_start = time.time()
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        feed_end = time.time()
        #print(f'Feed Forward Module 2 time {feed_end - feed_start:.6f} seconds')
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
    
        return x, mask



"""
NOTE (Ziwei):
This code is a modified version based on the icefall repository example.

Original Reference:
https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR/conformer_ctc
"""

class Icefall_vggsubsampling(nn.Module):
    """Trying to follow the setup described in the following paper:
    https://arxiv.org/pdf/1910.09799.pdf

    This paper is not 100% explicit so I am guessing to some extent,
    and trying to compare with other VGG implementations.

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' = T//4
    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct a VggSubsampling object.

        This uses 2 VGG blocks with 2 Conv2d layers each,
        subsampling its input by a factor of 4 in the time dimensions.

        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        super().__init__()

        cur_channels = 1
        layers = []
        block_dims = [32, 64]

        # The decision to use padding=1 for the 1st convolution, then padding=0
        # for the 2nd and for the max-pooling, and ceil_mode=True, was driven by
        # a back-compatibility concern so that the number of frames at the
        # output would be equal to:
        #  (((T-1)//2)-1)//2.
        # We can consider changing this by using padding=1 on the
        # 2nd convolution, so the num-frames at the output would be T//4.
        for block_dim in block_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.Conv2d(
                    in_channels=block_dim,
                    out_channels=block_dim,
                    kernel_size=3,
                    padding=0,
                    stride=1,
                )
            )
            layers.append(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )
            cur_channels = block_dim

        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(block_dims[-1] * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        start_time = time.time()
        x = x.unsqueeze(1)
        x = self.layers(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        end_time = time.time()
        print(f'1/4 k2 sub times{end_time - start_time:.6f} seconds')        
        '''
        NOTE (Ziwei): In icefall lens is lhotse format.
        See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32 
        I just follow CAT style
        
        '''
        lens_out = torch.div(torch.div(torch.div(lens - 1, 2) - 1, 2) - 1, 1).clamp(min=1).to(lens.dtype)
        return x, lens_out


class Icefall_relpositionalencoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(Icefall_relpositionalencoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)
    

class Icefall_conformerencoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
        normalize_before: whether to use layer_norm before the first block.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
    ) -> None:
        super(Icefall_conformerencoderLayer, self).__init__()
        self.self_attn = Icefall_relpositionmultiheadattention(d_model, nhead, dropout=0.0)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = Icefall_convolutionmodule(d_model, cnn_module_kernel)

        self.norm_ff_macaron = nn.LayerNorm(d_model)  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(d_model)  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """

        # macaron style feed forward module
        feed_model1_start = time.time()
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(src))
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)
        feed_model1_end = time.time()
        print(f'Feed Forward Model 1 time:{feed_model1_end - feed_model1_start:.6f} seconds')
        # multi-headed self-attention module
        attention_start = time.time()
        residual = src
        if self.normalize_before:
            src = self.norm_mha(src)
        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            src = self.norm_mha(src)
        attention_end = time.time()
        print(f'Multi-Headed Self-Attention Module time:{attention_end - attention_start:.6f} seconds')
        # convolution module
        conv_start = time.time()
        residual = src
        if self.normalize_before:
            src = self.norm_conv(src)
        src = residual + self.dropout(
            self.conv_module(src, src_key_padding_mask=src_key_padding_mask)
        )
        if not self.normalize_before:
            src = self.norm_conv(src)
        conv_end = time.time()
        print(f'Conformer Convolution Module time:{conv_end - conv_start:.6f} seconds')
        # feed forward module
        feed2_start = time.time()
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)
        feed2_end = time.time()
        print(f'Feed Forward Module 2 time{feed2_end - feed2_start:.6f} seconds')
        return src


class Icefall_relpositionmultiheadattention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super(Icefall_relpositionmultiheadattention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
            self.embed_dim,
            self.num_heads,
            self.in_proj.weight,
            self.in_proj.bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1).
                time1 means the length of query vector.

        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape
        assert n == 2 * time1 - 1
        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, time1, time1),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def multi_head_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        dropout_p: float,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
        training: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(
                3, dim=-1
            )

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                print(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            print(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + self.pos_bias_v).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k)  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p.transpose(-2, -1)
        )  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd)

        attn_output_weights = (
            matrix_ac + matrix_bd
        ) * scaling  # (batch, head, time1, time2)

        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, -1)

        assert list(attn_output_weights.size()) == [
            bsz * num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


class Icefall_convolutionmodule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        """Construct an ConvolutionModule object."""
        super(Icefall_convolutionmodule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Icefall_ConformerEncoder(nn.TransformerEncoder):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None
    ) -> None:
        super(Icefall_ConformerEncoder, self).__init__(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        output = src

        for mod in self.layers:
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output