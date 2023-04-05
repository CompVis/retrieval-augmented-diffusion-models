import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from ldm.util import default, exists
from ldm.modules.diffusionmodules.util import checkpoint, conv_nd
from ldm.modules.attention import FeedForward, zero_module

from rdm.modules.custom_clip.model import LayerNorm


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., causal=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.causal = causal
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.inner_dim = inner_dim
        self.context_dim = context_dim

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if self.causal:
            mask_value = -torch.finfo(sim.dtype).max
            i, j = sim.shape[-2:]
            r = torch.arange(i, device=x.device)
            mask = rearrange(r, 'i -> () i ()') < rearrange(r, 'j -> () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            sim.masked_fill_(mask, mask_value)
            del mask

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, causal=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, causal=causal)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, causal=causal if context_dim is None else False)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerBlockSingleAttention(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True, causal=False):
        super().__init__()
        self.attn = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, causal=causal)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


TRANSFORMER_BLOCKS = {"vanilla": BasicTransformerBlock}

SINGLE_ATTENTION_TRANSFORMER_BLOCKS = {"vanilla": BasicTransformerBlockSingleAttention}


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 dims=2, checkpoint=True, attn="vanilla",
                 num_norm_groups=32):

        super().__init__()
        assert attn in TRANSFORMER_BLOCKS.keys()
        assert type(context_dim) in [type(None), None, int, list], \
            f"context is of type {type(context_dim)} but should be None, int or list"
        if type(context_dim) != list: context_dim = [context_dim]*depth
        assert len(context_dim) == depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        assert dims in [2,3]
        self.dims = dims

        self.norm = Normalize(in_channels, num_norm_groups)

        self.proj_in = conv_nd(dims,
                               in_channels,
                               inner_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.transformer_blocks = nn.ModuleList(
            [TRANSFORMER_BLOCKS[attn](inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                      checkpoint=checkpoint)

                for d in range(depth)]
        )

        self.proj_out = zero_module(conv_nd(dims,
                                            inner_dim,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if type(context) != list:
            context = [context] * len(self.transformer_blocks)
        elif isinstance(context,list) and len(context) == 1:
            context *= len(self.transformer_blocks)

        if self.dims == 2:
            b, c, h, w = x.shape
        else:
            b, c, t, h, w = x.shape

        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.dims == 2:
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            x = rearrange(x, 'b c t h w -> b (t h w) c')  # TODO: add option for time-only attention
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.dims == 2:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            x = rearrange(x, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class RetrievalPatchTransformer(nn.Module):
    """
    Transformer block for data from the retrieval database.
    First add positional embeddings to all nns from a given patch for the respective position
    Then apply standard transformer action.
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 context_dim=None,
                 dropout=0.,
                 positional_encodings=False,
                 sequence_length=None,
                 residual=False,
                 checkpoint=False,
                 out_channels=None,
                 cross_attend=False,
                 causal=False,
                 continuous=True):
        # TODO enable gradient checkpoint, which, for some strange reason doesn't work yet
        super().__init__()
        if cross_attend: assert context_dim is not None
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.residual = residual
        self.checkpoint = checkpoint
        self.continuous = continuous

        if continuous:
            self.norm = LayerNorm(self.in_channels)
            self.proj_in = nn.Conv1d(self.in_channels,inner_dim,1)
        else:
            self.proj_in = nn.Embedding(self.in_channels, inner_dim)

        self.positional_encoding = None
        if positional_encodings:
            assert sequence_length is not None, 'Need sequence length for positional embedding'
            self.positional_encoding = nn.Parameter(torch.randn(inner_dim,sequence_length) / inner_dim ** 0.5)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,checkpoint=self.checkpoint, causal=causal)
             for d in range(depth)]
        )
        out_channels = default(out_channels, self.in_channels)
        self.proj_out = nn.Conv1d(inner_dim, out_channels, 1)


    def forward(self,x,context=None):

        # b, c , t = x.shape
        # t is here assumed to be n_patches * k_nns

        x_in = x
        if self.continuous:
            x = self.norm(x)
            x = rearrange(x, 'b t c -> b c t')
        x = self.proj_in(x)
        if not self.continuous:
            x = rearrange(x, 'b t c -> b c t')
        if self.positional_encoding is not None:
            x = x + self.positional_encoding[None, :, :x.shape[2]].type_as(x)

        x = rearrange(x, 'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x,context=context)
        x = rearrange(x, 'b t c -> b c t')
        x = self.proj_out(x)
        x = rearrange(x, 'b c t -> b t c')
        if self.residual:
            return x + x_in
        return x


class SimpleTransformer(nn.Module):
    """
    Transformer block
    First add positional embeddings to all nns from a given patch for the respective position
    Then apply standard transformer action.
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 positional_encodings=False,
                 sequence_length=None,
                 residual=False,
                 checkpoint=False,
                 out_channels=None,
                 causal=False,
                 continuous=True):
        # TODO enable gradient checkpoint, which, for some strange reason doesn't work yet
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.residual = residual
        self.checkpoint = checkpoint
        self.continuous = continuous

        if continuous:
            self.norm = LayerNorm(self.in_channels)
            self.proj_in = nn.Conv1d(self.in_channels,inner_dim,1)
        else:
            self.proj_in = nn.Embedding(self.in_channels, inner_dim)

        self.positional_encoding = None
        if positional_encodings:
            assert sequence_length is not None, 'Need sequence length for positional embedding'
            self.positional_encoding = nn.Parameter(torch.randn(inner_dim,sequence_length) / inner_dim ** 0.5)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlockSingleAttention(inner_dim, n_heads, d_head, dropout=dropout,checkpoint=self.checkpoint, causal=causal)
             for d in range(depth)]
        )
        out_channels = default(out_channels, self.in_channels)
        self.proj_out = nn.Conv1d(inner_dim, out_channels, 1)


    def forward(self, x, context=None):

        # b, c , t = x.shape
        # t is here assumed to be n_patches * k_nns

        x_in = x
        if self.continuous:
            x = self.norm(x)
            x = rearrange(x, 'b t c -> b c t')
        x = self.proj_in(x)
        if not self.continuous:
            x = rearrange(x, 'b t c -> b c t')
        if self.positional_encoding is not None:
            x = x + self.positional_encoding[None, :, :x.shape[2]].type_as(x)

        x = rearrange(x, 'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.proj_out(x)
        x = rearrange(x, 'b c t -> b t c')
        if self.residual:
            return x + x_in
        return x

class RetrievalTemporalTokenTransformerWrapper(RetrievalPatchTransformer):

    def __init__(self, input_vocab_size, in_channels, *args, **kwargs):
        super().__init__(in_channels, *args, **kwargs)
        self.tok_emb = nn.Embedding(input_vocab_size, in_channels)

    def forward(self, x, context=None):
        # expects a sequence of shape b n (h w), reshapes to b (n h w), embeds, reshapes to b n d again
        b, n, hw = x.shape
        x = rearrange(x, 'b n s-> b (n s)')
        x = self.tok_emb(x)
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=hw, d=self.in_channels, n=n)
        return super(RetrievalTemporalTokenTransformerWrapper, self).forward(x, context)


class EncoderDecoderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # custom implementation of x-transformers, todo


class GIT(nn.Module):
    """generative masked image transformer, optionally conditioned on r (continuous time-step)"""
    def __init__(self, input_vocab, output_vocab, embed_dim, n_layer, max_seq_len, n_heads,
                 d_head=None, dropout=0., r_conditional=True, attn="vanilla", context_dim=None,
                 checkpoint=False):
        super().__init__()

        self.tok_emb = nn.Embedding(input_vocab, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        #if type(context_dim) != list:
        #    context_dim = [context_dim] * n_layer

        if not d_head: assert embed_dim % n_heads == 0
        d_head = default(d_head, embed_dim // n_heads)

        if context_dim is not None:
            self.transformer_blocks = nn.ModuleList(
                [TRANSFORMER_BLOCKS[attn](embed_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                          checkpoint=checkpoint)
                 for d in range(n_layer)]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [SINGLE_ATTENTION_TRANSFORMER_BLOCKS[attn](embed_dim, n_heads, d_head, dropout=dropout,
                                                           # context_dim=context_dim[d],
                                                           checkpoint=checkpoint)
                 for d in range(n_layer)]
            )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.to_logits = nn.Linear(embed_dim, output_vocab, bias=False)
        self.r_conditional = r_conditional
        if self.r_conditional:
            self.to_r = nn.Linear(1, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, r=None, context=None):
        x = self.tok_emb(x)
        x = x + self.pos_emb[:, :x.shape[1], :]

        if r is not None:
            assert self.r_conditional
            r_emb = self.to_r(r)
            x = torch.cat((r_emb, x), 1)

        for block in self.transformer_blocks:
            if context is not None:
                x = block(x, context)
            else:
                x = block(x)

        x = self.ln_f(x)
        x = self.to_logits(x)
        if r is not None:
            x = x[:, r.shape[1]:]   # cut it off
        return x
