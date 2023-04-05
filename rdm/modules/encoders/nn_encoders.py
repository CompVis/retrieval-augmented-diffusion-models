import kornia
import torch.nn
import pytorch_lightning as pl
from einops import rearrange, repeat

from ldm.models.autoencoder import VQModel
from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.x_transformer import (AbsolutePositionalEmbedding, Encoder,
                                       always, exists)
from ldm.util import instantiate_from_config


class ClassicVQEncoder(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.decoder

    def forward(self, input, **kwargs):
        quant, diff, (_, _, ind) = super().encode(input)
        return quant, diff, ind

    def encode(self, x):
        quant, diff, (_, _, ind) = super().encode(x)
        return quant


class CodebookNNEncoder(VQModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.decoder

    def decode(self, h, force_not_quantize=False):
        raise NotImplementedError('Only encoder and quantizer initialized')

    def forward(self, input, **kwargs):
        quant, diff, (_, _, ind) = super().encode(input)
        return quant, diff, ind

    def encode(self, x):
        quant, diff, (_, _, ind) = super().encode(x)
        return ind


class SpatioTemporalNNEncoder(VQModel):

    def __init__(self, k, npatches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.npatches = npatches
        del self.decoder

    def decode(self, h, force_not_quantize=False):
        raise NotImplementedError('Only encoder and quantizer initialized')

    def encode(self, x):
        quant, diff, (_, _, ind) = super().encode(x)
        quant = rearrange(quant, '(b n k) c h w -> b (n k h w) c', n=self.npatches, k=self.k)
        ind = rearrange(ind, '(b s) -> b s', s=quant.shape[1])
        return quant, ind

    def forward(self, input, **kwargs):
        out = self.encode(input)
        quant, ind = out
        return quant, ind

class SpatioTemporalZNNEncoder(SpatioTemporalNNEncoder):
    '''
    Extracts codebook entry
    '''

    def encode(self, x):
        quant, ids = super().encode(x)
        return quant


class SpatioTemporalCodeNNEncoder(SpatioTemporalNNEncoder):
    '''
    Extracts code, i.e. subsequent transformer can learn embedding on its own
    '''

    def encode(self, x):
        quant, ids = super().encode(x)
        return ids


class TemporalNNCodeEncoder(VQModel):
    def __init__(self, k, npatches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.npatches = npatches
        del self.decoder

    def decode(self, h, force_not_quantize=False):
        raise NotImplementedError('Only encoder and quantizer initialized')

    def encode(self, x):
        quant, diff, (_, _, ind) = super().encode(x)
        #quant = rearrange(quant, '(b n k) c h w -> b (n k h w) c', n=self.npatches, k=self.k)
        ind = rearrange(ind, '(b n k h w) -> b (n k) (h w)', h=quant.shape[2], w=quant.shape[3],
                        k=self.k, n=self.n_patches)
        return ind

    def forward(self, input, **kwargs):
        return self.encode(input)


class TemporalNNZEncoder(VQModel):
    def __init__(self, k, npatches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.npatches = npatches
        del self.decoder

    def decode(self, h, force_not_quantize=False):
        raise NotImplementedError('Only encoder and quantizer initialized')

    def encode(self, x):
        quant, diff, (_, _, ind) = super().encode(x)
        quant = rearrange(quant, '(b n k) c h w -> b (n k) (h w c)', n=self.npatches, k=self.k)
        return quant

    def forward(self, input, **kwargs):
        return self.encode(input)


class CLIPEmbeddingReshaper(object):
    def __call__(self, x):
        nn_patches = rearrange(x, 'b n k d -> b (n k) d').to(torch.float)
        return nn_patches


class IdentityEncoder(torch.nn.Module):
    def __init__(self, to_float32=True):
        super().__init__()
        self.to_float32 = to_float32

    def __call__(self, x):
        return x

    def encode(self, x):
        out = self(x)
        if self.to_float32:
            out = out.to(torch.float)
        return out


class SpatioTemporalConcatEncoder(VQModel):
    def __init__(self, k, npatches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.npatches = npatches
        del self.decoder

    def decode(self, h, force_not_quantize=False):
        raise NotImplementedError('Only encoder and quantizer initialized')

    def encode(self, x):
        quant, diff, (_, _, ind) = super().encode(x)
        quant = rearrange(quant, '(b n k) c h w -> b (n k c) h w', n=self.npatches, k=self.k)
        ind = rearrange(ind, '(b n k h w) -> b (n k) h w', h=quant.shape[-2],
                        w=quant.shape[-1], b=quant.shape[0],n=self.npatches,k=self.k)
        return quant, ind

    def forward(self, input, **kwargs):
        out = self.encode(input)
        quant, ind = out
        return quant, ind


class SpatioTemporalConcatZNNEncoder(SpatioTemporalConcatEncoder):
    '''
    Extracts codebook entry
    '''
    def encode(self, x):
        quant, ids = super().encode(x)
        return quant


class SpatioTemporalConcatCodeNNEncoder(SpatioTemporalConcatEncoder):
    '''
    Extracts code, i.e. subsequent transformer can learn embedding on its own
    '''
    def encode(self, x):
        quant, ids = super().encode(x)
        return ids

class VQConcatNNEncoder(VQModel):

    def __init__(self, k, npatches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.npatches = npatches
        del self.decoder

    def encode(self,x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        h = rearrange(h,'(b n k) c h w -> (n k) b c h w',n=self.npatches,k=self.k)
        # make list for concatenation
        return list(h)


class PixelNNEncoderUnetConcatenator(torch.nn.Module):
    def __init__(self, k, encoder_cfg, backbone_cfg, npatches=1, ):
        super().__init__()
        self.k = k
        self.npatches = npatches
        self.encoder = instantiate_from_config(encoder_cfg)
        self.backbone = instantiate_from_config(backbone_cfg)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        # parallel for each (n k)
        x = self.encoder(x)
        # merge neigbors in channels axis
        x = rearrange(x, '(b n k) c h w -> b (n k c) h w', n=self.npatches, k=self.k)
        # pass trhough unet and produce spatial output
        x = self.backbone(x)
        return x

class DimensionStackerVQEncoder(VQModel):

    def __init__(self,n_patches, k,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_patches = n_patches
        self.k = k
        del self.decoder

    def encode(self, x):
        quant, _, _ = super().encode(x)
        #quant = rearrange(quant, '(b n k) c h w -> b (n k h w) c', n=self.npatches, k=self.k)
        c = rearrange(quant, '(b n k) c h w -> b (n k) c h w', n=self.n_patches , k=self.k)
        c = rearrange(c, 'b p c h w -> b (h w) (p c)')
        return c


    def forward(self,x, **kwargs):
        return self.encode(x)


class DummyEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def encode(self, x):
        return x

    def forward(self, x):
        return x


class VQGANAggregator(pl.LightningModule):
    '''
    Aggregates all information of each individual nn per patch in a bert-style
    classifcation output and constructs an output sequence by using each output
    as individual sequence element. Learns an embedding given VQ codewords
    '''
    def __init__(self,
               k,
               n_patches,
               num_tokens,
               embed_dim,
               seq_len,
               context_dim,
               n_transformer_layers,
               use_pos_emb=True,
               use_checkpoint=False,
               **kwargs):
        super().__init__()
        # add 1 to seq len because of cls token
        self.sequence_length = seq_len+1
        self.use_checkpoint = use_checkpoint
        self.n_patches = n_patches
        self.k = k
        self.token_emb = torch.nn.Embedding(num_tokens, embed_dim)
        self.pos_emb = AbsolutePositionalEmbedding(embed_dim, self.sequence_length) if use_pos_emb else always(0)
        # learnable cls token
        self.cls_token = torch.nn.Parameter(0.02 * torch.randn((embed_dim,),))
        # transformer
        self.transformer = Encoder(dim=embed_dim,depth=n_transformer_layers,
                                   **kwargs)
        #head
        self.norm = torch.nn.LayerNorm(embed_dim)
        # head to output dim (which is the context_dim of the Decoder)
        self.head = torch.nn.Linear(embed_dim,context_dim)


    def _forward(self,x):

        # add class token
        cls = repeat(self.cls_token,'d -> b 1 d',b=x.shape[0])
        x = torch.cat([cls,x],dim=1)
        x+=self.pos_emb(x)
        # transformer action
        x = self.transformer(x)
        # TODO norm? y/n, if yes where?
        x = self.norm(x)
        x = self.head(x[:,0])
        # rearrange
        return rearrange(x,'(b n k) d -> b (n k) d',n=self.n_patches,k=self.k)

    def forward(self,ind):
        # ind has shape b (n k) s
        ind = rearrange(ind, 'b n h w -> (b n) (h w)')

        x = self.token_emb(ind)
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )


class ContinuousVQGANAggregator(pl.LightningModule):
    '''
    Aggregates all information of each individual nn per patch in a bert-style
    classifcation output and constructs an output sequence by using each output
    as individual sequence element. Directly acts on VQ encodings
    '''
    def __init__(self,
               k,
               n_patches,
               embed_dim,
               seq_len,
               context_dim,
               n_transformer_layers,
               input_dim=None,
               use_pos_emb=True,
               use_checkpoint=False,
               **kwargs):
        super().__init__()
        # add 1 to seq len because of cls token
        self.sequence_length = seq_len+1
        self.use_checkpoint = use_checkpoint
        self.n_patches = n_patches
        self.k = k

        self.proj_in = torch.nn.Linear(input_dim,embed_dim) if exists(input_dim) else torch.nn.Identity()
        self.pos_emb = AbsolutePositionalEmbedding(embed_dim, self.sequence_length) if use_pos_emb else always(0)
        # learnable cls token
        self.cls_embedding = torch.nn.Parameter(0.02 * torch.randn((embed_dim,),))
        # transformer
        self.transformer = Encoder(dim=embed_dim,depth=n_transformer_layers,
                                   **kwargs)
        #head
        self.norm = torch.nn.LayerNorm(embed_dim)
        # head to output dim (which is the context_dim of the Decoder)
        self.head = torch.nn.Linear(embed_dim,context_dim)


    def _forward(self,x,context=None):



        if x.ndim == 4:
            # x is assumed to have spatial dimensions with the channel axis first, as eg. vqgan outputs
            x = rearrange(x,'b c h w -> b (h w) c')

        x = self.proj_in(x)
        # add class embedding
        cls = repeat(self.cls_embedding,'d -> b 1 d',b=x.shape[0])
        x = torch.cat([cls,x],dim=1)
        x+=self.pos_emb(x)
        # transformer action
        x = self.transformer(x,context)
        # TODO norm? y/n, if yes where?
        x = self.norm(x)
        out = self.head(x[:,0])
        # rearrange
        return out

    def forward(self,x,context=None,k=None,n_patches=None):

        args = (x,)

        if k is None:
            k = self.k

        if n_patches is None:
            n_patches = self.n_patches

        if context is not None:
            args += (context,)


        out = checkpoint(
            self._forward, args, self.parameters(), self.use_checkpoint
        )

        return rearrange(out, '(b n k) d -> b (n k) d', n=n_patches, k=k)

class VQGANNNAttender(pl.LightningModule):

    def __init__(self,spatial_condenser_cfg,nn_attender_cfg):
        super().__init__()
        self.spatial_condenser = instantiate_from_config(spatial_condenser_cfg)
        self.nn_attender = instantiate_from_config(nn_attender_cfg)

    def forward(self, x, context=None, n=None, k=None):
        # input shape is (b n k) c h w
        x = rearrange(x,'b c h w -> b (h w) c')
        # extract unified representation for all spatial dimensions of each element
        context_ = None
        if context is not None:
            if not n:
                n = self.spatial_condenser.n_patches
            if not k:
                k = self.spatial_condenser.k

            context_ = repeat(context,'b s c -> (b r) s c',r=n*k)

        x = self.spatial_condenser(x,context_,k=k,n_patches=n) # input shape: (b n k) (h w) c, output shape b (n k) d
        # attention between different nns of different patches
        return self.nn_attender(x,context) # input shape b (n k) d , output shape b (n k) context_dim
