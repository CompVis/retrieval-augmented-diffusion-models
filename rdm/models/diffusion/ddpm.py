"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
import pickle

import kornia
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from ldm.models.autoencoder import (AutoencoderKL, IdentityFirstStage,
                                    VQModelInterface)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config, isimage, ismap, log_txt_as_img
from ldm.modules.ema import LitEma

from rdm.util import ischannellastimage
from rdm.models.diffusion.ddim import DDIMSampler

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key, concat_dim=1):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.retro_mode = False
        self.concat_dim = concat_dim
        if conditioning_key == "retro_only":
            self.retro_mode = True
            conditioning_key = None
            print(f"{self.__class__.__name__}: Instantiating RETRO-only mode, "
                  f"i.e. used for unconditional training with RETRO augmentation")
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']
        self.wrapper_conditioning_key = self.conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None and not self.retro_mode:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x]+c_concat, dim=self.concat_dim)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn' or self.retro_mode:
            cc = torch.cat(c_crossattn, self.concat_dim)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=self.concat_dim)
            cc = torch.cat(c_crossattn, self.concat_dim)

            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out

class RETRODiffusionWrapper2(pl.LightningModule):

    def __init__(self, diffusion_wrapper, conditioning_key = 'crossattn'):
        super().__init__()
        assert isinstance(diffusion_wrapper, DiffusionWrapper), f'{self.__class__.__name__} requires first parameter to be of type "DiffusionWrapper"'
        self.diffusion_wrapper = diffusion_wrapper
        self.retro_conditioning_key = conditioning_key

        if self.diffusion_wrapper.conditioning_key is None:
            print(f'WARNING: {self.__class__.__name__} changing conditioning_key of DiffusionWrapper from {self.diffusion_wrapper.conditioning_key} to {conditioning_key}')
            self.diffusion_wrapper.conditioning_key = conditioning_key
            self.conditioning_key = None
        elif self.diffusion_wrapper.conditioning_key != conditioning_key:
            print(f'WARNING: {self.__class__.__name__} changing conditioning_key of DiffusionWrapper from {self.diffusion_wrapper.conditioning_key} to "hybrid"')
            self.diffusion_wrapper.conditioning_key = 'hybrid'
            self.conditioning_key = self.diffusion_wrapper.conditioning_key
        else:
            self.conditioning_key = self.diffusion_wrapper.conditioning_key

        self.wrapper_conditioning_key = self.diffusion_wrapper.conditioning_key


    def forward(self, *args, **kwargs):
        return self.diffusion_wrapper(*args,**kwargs)

class RETRODiffusionWrapper(pl.LightningModule):
    # TODO: discuss
    """
    for now, drop options regarding old code and only use cross-attention
    """

    def __init__(self, diffusion_wrapper,concat=False):
        super().__init__()
        self.concat = concat
        if concat:
            print(f'WARNING: {self.__class__.__name__} is concatenating conditionings in sequence dimension. Assuming same embedding dimension')
            self.diffusion_model = diffusion_wrapper
        else:
            self.diffusion_model = diffusion_wrapper.diffusion_model
        self.conditioning_key = diffusion_wrapper.conditioning_key
        self.wrapper_conditioning_key = diffusion_wrapper.conditioning_key
        print(f"{self.__class__.__name__}: Wrapping diffusion model for RETRO training. "
              f"For multimodal data, conditionings will be chained in a list "
              f"and all fed into the 'SpatialTransformer' via different cross-attention "
              f"blocks.")

    def forward(self, x, t, c_crossattn: list = None):
        key = 'c_crossattn' if self.concat else 'context'
        out = self.diffusion_model(x, t, **{key:c_crossattn})
        return out


class MinimalRETRODiffusion(LatentDiffusion):
    """main differences to base class:
        - dataloading to build the conditioning
        - concat the base conditioning (e.g. text) and the new retro-conditionings
        - maybe adopt log_images
    """
    def __init__(self, k_nn, query_key, retrieval_encoder_cfg, nn_encoder_cfg=None, query_encoder_cfg=None,
                 nn_key='retro_conditioning', retro_noise=False, retrieval_cfg=None,retro_conditioning_key=None,
                 learn_nn_encoder=False, nn_memory=None,
                 n_patches_per_side = 1, resize_patch_size=None,
                 searcher_path=None, retro_concat=False,
                 p_uncond=0., guidance_vex_shape=None,
                 *args, **kwargs):
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])
        use_ema = kwargs.pop('use_ema',True)
        unet_config = kwargs.get('unet_config',{})
        super().__init__(*args, **kwargs)

        self.k_nn = k_nn  # number of neighbors to retrieve
        self.query_key = query_key  # e.g. "clip_txt_emb"
        self.nn_key = nn_key
        # backwards compat
        if retro_conditioning_key is not None:
            self.model = RETRODiffusionWrapper2(self.model, retro_conditioning_key)
        else:
            self.model = RETRODiffusionWrapper(self.model,concat=retro_concat)  # adopts the forward pass to retro-style cross-attn training
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.searcher_path = searcher_path

        self.use_memory = nn_memory is not None and os.path.isfile(nn_memory)
        if self.use_memory:
            assert os.path.isfile(nn_memory) and nn_memory.endswith('.p')
            print(f'Loading nn_memory from "{nn_memory}"')
            with open(nn_memory,'rb') as f:
                nn_data = pickle.load(f)
            self.register_buffer('nn_memory',torch.tensor(nn_data['nn_memory'],dtype=torch.int),persistent=False)
            print(f'Loaded nn_memory of size {self.nn_memory.shape[0]}')
            self.id_count = nn_data['id_count']

        self.retriever = None
        self.learn_nn_encoder = learn_nn_encoder
        self.init_nn_encoder(nn_encoder_cfg)  # TODO attention while restoring, do not want to overwrite
        self.resize_nn_patch_size = resize_patch_size
        self.init_retriever(retrieval_cfg)
        # most likely a transformer, can apply masking within this models' forward pass
        self.conditional_retrieval_encoder = query_encoder_cfg is not None
        if self.conditional_retrieval_encoder and 'cross_attend' not in retrieval_encoder_cfg.params:
            print(
                f'WARNING: intending to train query conditioned retrieval encoder without cross attention, adding option to cfg...')
            retrieval_encoder_cfg.params['cross_attend'] = True
        self.retrieval_encoder = instantiate_from_config(
            retrieval_encoder_cfg)  # TODO attention while restoring, do not want to overwrite
        self.init_query_encoder(query_encoder_cfg)

        self.retro_noise = retro_noise


        self.n_patches_per_side = n_patches_per_side

        self.use_retriever_for_retro_cond = not self.retriever is None and not self.use_memory

        self.p_uncond = p_uncond
        # if p_uncond > 0:
        if guidance_vex_shape is None:
            guidance_vex_shape = (unet_config.params.context_dim,)
            print(f'Setting guiding vex shape to (,) (assuming clip nn encoder)')


        self.get_unconditional_guiding_vex(tuple(guidance_vex_shape))
        # else:
        #     ignore_keys += ['unconditional_guidance_vex']

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_query_encoder(self, cfg):
        if not self.conditional_retrieval_encoder:
            return
        self.query_encoder = instantiate_from_config(cfg)
        print(f'Using {self.query_encoder.__class__.__name__} as query encoder.')


    def init_nn_encoder(self, cfg):
        if not cfg:
            self.nn_encoder = None
            self.resize_nn_patches = False
            return
        self.resize_nn_patches = cfg.params.pop('resize_nn_patches', False)
        if cfg == '__is_first_stage__':
            self.learn_nn_encoder = False
            print("Using first stage also as nn_encoder.")
            self.nn_encoder = self.first_stage_model
            self.resize_nn_patches = True
        else:
            self.nn_encoder = instantiate_from_config(cfg)
            additional_info = 'LEARNABLE' if self.learn_nn_encoder else 'FIXED'
            print(f'Loading {additional_info} nn_encoder of type {self.nn_encoder.__class__.__name__}')
            if not self.learn_nn_encoder:
                self.nn_encoder.train = disabled_train
                for param in self.nn_encoder.parameters():
                    param.requires_grad = False

        cfg.params['resize_nn_patches'] = self.resize_nn_patches

    @rank_zero_only
    def train_searcher(self):
        print("training searcher...")
        self.retriever.train_searcher()
        print("done training searcher")


    @rank_zero_only
    def init_retriever(self, cfg):
        if not cfg:
            self.retriever = None
            return
        # this is the nearest neighbor searcher
        self.retriever = instantiate_from_config(cfg)
        self.retriever.train = disabled_train
        for param in self.retriever.retriever.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def get_nn_and_encoding(self,query, return_patches=False,
                            k_nn=None, n_patches_per_side=None,
                            return_query_patches=False):
        # check if searcher is trained and train, if required
        if self.retriever.searcher is None:
            self.train_searcher()

        output=dict()
        n_ptch = self.n_patches_per_side if n_patches_per_side is None else n_patches_per_side
        k_nn = self.k_nn if k_nn is None else k_nn
        if not isimage(query):
            query = rearrange(query,'b h w c -> b c h w')
        patch_side_len = query.shape[-1] // n_ptch
        queries = []

        # create patches
        for i in range(n_ptch):
            for j in range(n_ptch):
                queries.append(query[...,
                               i*patch_side_len:(i+1)*patch_side_len,
                               j*patch_side_len:(j+1)*patch_side_len])

        if return_query_patches:
            img_size = self.first_stage_model.encoder.resolution
            qp = self.resize_img_batch(rearrange(torch.stack(queries,dim=1),'b n c h w -> (b n) c h w'),
                                        size=img_size)
            output['query_patches'] = rearrange(qp, '(b n) c h w -> b n () c h w',n=n_ptch**2)

        queries = rearrange(torch.stack(queries,dim=1),'b n c h w -> (b n) c h w')

        query_embeddings = self.retriever.retriever(queries)  # identity or some model like CLIP.

        query_embeddings = query_embeddings.detach().cpu().numpy()  # todo: or should we assume that the query_embedder produces numpy arrays?

        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis]
        nns, distances = self.retriever.searcher.search_batched(query_embeddings, final_num_neighbors=k_nn)

        # !!! TODO: implement different ways for obtaining out_embeddings. Could also learn a codebook or use images directly, ...
        out = self.retriever.data_pool['embedding'][nns]
        out = rearrange(out,'(b n) k d -> b n k d',n=n_ptch**2)
        out = torch.Tensor(out).to(self.device)
        output[self.nn_key] = out

        if return_patches:
            patches = self.retriever.get_nn_patches(nns)
            output["image_patches"] = patches


        if self.nn_encoder is not None:
            in_shape = patches.shape
            output[self.nn_key] = self.encode_with_fixed_nn_encoder(rearrange(patches,'b n k h w c -> (b n k) c h w'),
                                                                    in_shape)

        return output

    def resize_img_batch(self,img_batch,size):
        return kornia.geometry.resize(img_batch,size=(size,size),align_corners=True)

    @torch.no_grad()
    def encode_with_fixed_nn_encoder(self,nns2encode,shape = None):
        if self.resize_nn_patches:
            resize_size = self.resize_nn_patch_size if self.resize_nn_patch_size else self.first_stage_model.encoder.resolution
            nns2encode = self.resize_img_batch(nns2encode, resize_size)
        if isinstance(self.nn_encoder, VQModelInterface):
            # for backwards compatibility
            if self.model.wrapper_conditioning_key in ['concat'] or (self.model.wrapper_conditioning_key == 'hybrid' and self.model.retro_conditioning_key == 'concat'):
                nn_encodings = self.nn_encoder.encode(nns2encode)
            else:
                assert shape is not None, 'Need to give \'em a shape'
                bs, nptch, k = shape[:3]
                nn_encodings = self.nn_encoder.encode(nns2encode).reshape(bs, nptch * k, -1)
        else:
            # NNEnoder is assumed to do reshaping
            nn_encodings = self.nn_encoder.encode(nns2encode)

        return nn_encodings


    @torch.no_grad()
    def get_retro_conditioning(self, batch, return_patches=False, bs=None,
                               use_learned_nn_encoder=False,k_nn=None, use_retriever=None):
        """
        given x, compute its nearest neighbors via the self.retriever module
        TODO: move this completely to dataloader
        """
        if k_nn is None:
            k_nn = self.k_nn
        output = dict()
        if bs is not None:
            batch = {key: batch[key][:bs] for key in batch}

        if use_retriever is None:
            use_retriever = self.use_retriever_for_retro_cond

        if not use_retriever or self.nn_encoder is not None:

            # retro conditioning is expected to have shape  (bs,n_query_patches,k,embed_dim)
            if self.nn_encoder is None:
                # use retriever embeddings
                nns = batch[self.nn_key]
                # reshape appropriately for transformer

                output[self.nn_key] = rearrange(nns, 'b n k d -> b (n k) d').to(torch.float)
            else:
                # use nn_patches with defined retrieval embedder model
                assert self.nn_encoder.device == self.device
                # bs, nptch, k = batch['nn_patches'].shape[:3]
                nn_patches = rearrange(batch['nn_patches'], 'b n k h w c -> (b n k) c h w').to(self.device).to(
                    torch.float)
                if not self.learn_nn_encoder or use_learned_nn_encoder:
                    output[self.nn_key] = self.encode_with_fixed_nn_encoder(nn_patches,batch['nn_patches'].shape)
                # else:
                #     raise AttributeError(f'When learn_nn_encoder is enabled, it must not be called from inside {self.__class__.__name__}.get_retro_conditioning()')
                # TODO implement?
                # if self.use_memory and self.training:
                #     raise NotImplementedError()

            if return_patches and 'nn_patches' in batch:
                output['image_patches'] = rearrange(batch['nn_patches'], 'b n k h w c -> (b n) k h w c')

        else:
            query = batch[self.query_key]
            # if bs is not None: query = query[:bs]
            output = self.get_nn_and_encoding(query,return_patches)

        return output  # as a sequence of neighbor representations, sorted by increasing distance.

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        r = self.get_retro_conditioning(batch,return_patches=self.learn_nn_encoder)
        if self.p_uncond > 0.:
            mask = torch.distributions.Bernoulli(torch.full((x.shape[0],),self.p_uncond)).sample().to(self.device).bool()
            uncond_signal = self.get_unconditional_conditioning(shape=r[self.nn_key].shape)
            r[self.nn_key] = torch.where(repeat(mask,'b -> b 1 1 '), uncond_signal,r[self.nn_key])
        loss = self(x, c, r)
        return loss



    def forward(self, x, c, r, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        qs = {}

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        noise = torch.randn_like(x)
        if self.conditional_retrieval_encoder:
            x_noisy = self.q_sample(x, t, noise=noise)
            q = self.query_encoder(x_noisy)
            qs = {'context': q}

        if self.learn_nn_encoder:
            assert self.nn_encoder is not None, 'If you wanna learn nn_encoders, please define such a thing :)'
            # NNEnoder is assumed to do reshaping
            r[self.nn_key] = self.nn_encoder.encode(rearrange(r['image_patches'],'(b n) k h w c -> (b n k) c h w',
                                                              b=x.shape[0],n= self.n_patches_per_side**2,k=self.k_nn).to(self.device).to(torch.float))

        r_enc = self.retrieval_encoder(r[self.nn_key], **qs)  # can also be initialized to the identity
        if self.retro_noise:
            r_enc = self.q_sample(r_enc,t)


        if c is not None:
            if self.model.wrapper_conditioning_key == 'hybrid':
                non_retro_ck = 'crossattn' if self.model.retro_conditioning_key == 'concat' else 'concat'
                c = {'c_'+non_retro_ck: c if isinstance(c,list) else [c],
                     'c_'+self.model.retro_conditioning_key: r_enc if isinstance(r_enc,list) else [r_enc]}
            else:
                c = [r_enc, c]
            # c = torch.cat((r_enc, c), dim=1)
            # TODO: make this configurable, maybe introduce additional module here
        else:
            if not isinstance(r_enc,list):
                c = [r_enc]
            else:
                c = r_enc

        return self.p_losses(x, c, t, noise=noise, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.wrapper_conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # if not isinstance(cond, list):
        #     cond = [cond]
        x_recon = self.model(x_noisy, t, **cond)
        return x_recon

    @rank_zero_only
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, display_patches=True, sample_nns=False,memsize=None, use_weights=False,
                   unconditional_guidance_scale=1.,**kwargs):
        if memsize is None:
            memsize = [100]
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch,
                                           self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)

        # retrieval-specific hacks
        rdir = self.get_retro_conditioning(batch, return_patches=display_patches, bs=N,
                                           use_learned_nn_encoder=self.learn_nn_encoder)
        display_patches &= 'image_patches' in rdir
        if display_patches:
            r, img_patches = rdir[self.nn_key], rdir["image_patches"]
        else:
            r = rdir[self.nn_key]

        if self.conditional_retrieval_encoder:
            qs = {'context': self.query_encoder(z)}
        else:
            qs = {}

        r_enc = self.retrieval_encoder(r, **qs)

        if c is not None:
            if self.model.wrapper_conditioning_key == 'hybrid':
                # this case can only occur if RETRODiffusionWrapper2 is used
                non_retro_ck = 'crossattn' if self.model.retro_conditioning_key == 'concat' else 'concat'
                c = {'c_' + non_retro_ck: c if isinstance(c, list) else [c],
                     'c_' + self.model.retro_conditioning_key: r_enc if isinstance(r_enc, list) else [r_enc]}
            else:
                c = [r_enc, c]
            # c = torch.cat((r_enc, c), dim=1)
            # TODO: make this configurable, maybe introduce additional module here
        else:
            if not isinstance(r_enc,list):
                c = [r_enc]
            else:
                c = r_enc

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        if display_patches:
            # plot neighbors
            # shape of img_patches: b n h w c
            grid = rearrange(img_patches, 'b n h w c -> (b n) c h w')
            grid = make_grid(grid, nrow=img_patches.shape[1], normalize=True)
            log["neighbors"] = 2. * grid - 1.

        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta,**kwargs)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True,**kwargs)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N], mask=mask,**kwargs)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N], mask=mask,**kwargs)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        # only test this for for one patch per query and for non query-condiotional retrieval encoder
        # TODO enable also for query conditional model
        if sample_nns and self.retriever is not None and batch[self.nn_key].shape[
            1] == 1 and not self.conditional_retrieval_encoder and self.cond_stage_model is None:
            if self.use_memory:
                for maxsize in tqdm(memsize,total=len(memsize),desc='Unconditional sampling for different memory sizes'):
                    out_r_data = self.sample_from_rdata(N, ddim=use_ddim, return_nns=self.retriever.load_patch_dataset,
                                                        ddim_steps=ddim_steps, eta=ddim_eta, memsize=maxsize, use_weights=use_weights)
                    if 'batched_nns' in out_r_data:
                        del out_r_data['batched_nns']
                    out_r_data = {key+f'@memsize={maxsize}': out_r_data[key] for key in out_r_data}
                    log.update(out_r_data)

                    if unconditional_guidance_scale > 1.:
                        out_r_guided = self.sample_from_rdata(N, ddim=use_ddim, return_nns=self.retriever.load_patch_dataset,
                                                        ddim_steps=ddim_steps, eta=ddim_eta, memsize=maxsize, use_weights=use_weights,
                                                              unconditional_guidance_scale=unconditional_guidance_scale)

                        out_r_guided = {key+f'@memsize={maxsize}-cfg_s{unconditional_guidance_scale}': out_r_guided[key] for key in out_r_guided}
                        log.update(out_r_guided)

            else:
                out_sampled_rdata = self.sample_from_rdata(N, ddim=use_ddim, return_nns=self.retriever.load_patch_dataset,
                                                           ddim_steps=ddim_steps, eta=ddim_eta,)
                log.update(out_sampled_rdata)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    # @torch.no_grad()
    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
    #     # only for very first batch
    #     r = self.get_retro_conditioning(batch,
    #                                         return_patches=self.learn_nn_encoder)
    #     self.get_unconditional_guiding_vex(r[self.nn_key].shape)

    def get_unconditional_guiding_vex(self,vector_shape):
        # if not hasattr(self, 'unconditional_guidance_vex'):
            # bs, vector_shape = shape[0], shape[1:]


        print('Initializing unconditional guidance vector')
        # if self.p_uncond == 0.:
        init_vec = torch.randn(vector_shape, device=self.device)
        # else:
        #     init_vec = torch.zeros((vector_shape,),device=self.device)
        if self.p_uncond == 0.:
            self.register_buffer('unconditional_guidance_vex', init_vec, persistent=True)
        else:
            self.register_parameter('unconditional_guidance_vex', torch.nn.Parameter(init_vec))

    @torch.no_grad()
    def get_unconditional_conditioning(self, shape, unconditional_guidance_label=None,k_nn=None, ignore_knn=False):

        if k_nn is None:
            k_nn = self.k_nn

        bs, vector_shape = shape[0], shape[-1]

        if not hasattr(self, 'unconditional_guidance_vex'):
            self.get_unconditional_guiding_vex((vector_shape,))

        if unconditional_guidance_label is not None:
            # unconditional guidance label defines norm of vector
            uncond_signal = self.unconditional_guidance_vex / torch.linalg.norm(self.unconditional_guidance_vex.flatten()) * unconditional_guidance_label
            # TODO dirty hack --> resolve
            if uncond_signal.shape[0] != self.k_nn and not ignore_knn:
                uncond_signal = torch.stack([uncond_signal] * k_nn, dim=0)

            uncond_signal = torch.stack([uncond_signal]*bs,dim = 0)
        else:
            uncond_signal = torch.stack([self.unconditional_guidance_vex]*bs,dim = 0)

        print(uncond_signal.shape)

        return uncond_signal

    @torch.no_grad()
    def sample_with_query(self, query, cond=None, bs=None, k_nn=None,
                          unconditional_guidance_scale=1., unconditional_guidance_label=None,
                          unconditional_retro_guidance_label=None, return_nns=False,
                          n_reps=None, query_embedded=False, example_maps=None,
                          visualize_nns=True, omit_query=False, normalize=False,
                          **kwargs):
        from rdm.modules.encoders.nn_encoders import (VQGANAggregator,
                                                      VQGANNNAttender)
        if not query_embedded:
            assert query.ndim in [3, 4], 'User defined query for sampling has to be an image or of batch of images'
        if self.retriever is not None and self.retriever.searcher is None:
            self.train_searcher()

        # TODO currently only supported for query as 1 patch
        if bs is None:
            if cond is not None:
                bs = cond.shape[0] if isinstance(cond, torch.Tensor) else len(cond)
            else:
                bs = 1

        if query.ndim == 3 and isimage(query[None]):
            query = torch.stack([query] * bs, dim=0)

        elif query.ndim == 4 and query.shape[0] == 1:
            query = repeat(query, '1 h w c -> b h w c', b=bs)

        elif isinstance(query, str):
            query = [query] * bs

        elif query_embedded and query.shape[0] == 1:
            query = repeat(query,'1 c -> b c', b = bs)

        is_caption = isinstance(query, list)

        assert ischannellastimage(query) or is_caption or query_embedded

        if k_nn is None:
            k_nn = self.k_nn

        print(query.shape)

        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()

        print(f'Query shape is {query.shape}')
        nn_dict = self.retriever.search_k_nearest(query,visualize=visualize_nns, k=k_nn,
                                                  is_caption=is_caption,
                                                  query_embedded=query_embedded)

        out = dict()


        if self.nn_encoder is not None:
            nn_patches = nn_dict['nn_patches']  # shape is (b k h w c)
            # insert query as first nn
            if isinstance(query, np.ndarray):
                query = torch.from_numpy(query).to(self.device).float()

            if n_reps is not None:
                nn_patches = torch.stack([query] * n_reps, dim=1)
            else:
                nn_patches = torch.cat([query[:, None], nn_patches[:, :k_nn - 1]], dim=1)

            print(nn_patches.shape)
            nn_patches = rearrange(nn_patches, 'b k h w c -> (b k) c h w').to(self.device)
            if return_nns:
                out['retro_nns'] = make_grid(nn_patches, nrow=k_nn if n_reps is None else n_reps, normalize=True)
                # out['retro_nns'] = 2.* out['retro_nns'] - 1.
            retro_cond = self.encode_with_fixed_nn_encoder(nn_patches).float()

        else:
            q_emb = torch.from_numpy(nn_dict['q_embeddings'])
            r_emb = torch.from_numpy(nn_dict['embeddings'])
            if normalize:
                q_emb = q_emb / q_emb.norm(dim=-1,keepdim=True)
                r_emb = r_emb / r_emb.norm(dim=-1,keepdim=True)
            # use the query embedding as first neighbour
            if example_maps is not None:
                retro_cond = torch.cat([q_emb[:, None], repeat(example_maps[:,None],
                                                               'b 1 c -> b k c',k=k_nn-1)], dim=1).to(self.device).float()  # shape is (b k 512) (for clip retriever model)
                if n_reps is not None:
                    retro_cond = repeat(torch.stack([q_emb,example_maps],dim=1),'b n c -> b (n r) c',r=n_reps // 2).to(self.device).float()
            else:
                if omit_query:
                    retro_cond = r_emb.to(self.device).float()
                else:
                    retro_cond = torch.cat([q_emb[:, None], r_emb[:, :k_nn - 1]], dim=1).to(self.device).float()  # shape is (b k 512) (for clip retriever model)
                if n_reps is not None:
                    retro_cond = torch.cat([retro_cond] * n_reps, dim=1)
            if return_nns:
                if n_reps is None and not query_embedded:
                    nn_patches = nn_dict['nn_patches'][:, :k_nn - 1]
                    if not omit_query:
                        if isinstance(query,np.ndarray):
                            query =torch.from_numpy(query).type_as(nn_patches)

                        query = kornia.geometry.resize(rearrange(query,'b h w c -> b c h w'),size=nn_patches.shape[-3:-1])
                        query = rearrange(query,'b c h w -> b h w c')
                        print(query.shape, nn_patches.shape)
                        nn_patches = torch.cat([query[:, None], nn_patches[:, :k_nn - 1]], dim=1)
                elif n_reps is None:
                    if omit_query:
                        nn_patches = nn_dict['nn_patches'][:, :k_nn]
                    else:
                        nn_patches = nn_dict['nn_patches'][:, :k_nn - 1]
                elif n_reps is not None and not query_embedded:
                    nn_patches = torch.stack([query] * n_reps, dim=1)

                # if caption based and caption shall be repeated, there are no visual neighbors
                if not (query_embedded and n_reps is not None):
                    if n_reps is None:
                        n_per_row = nn_patches.shape[1]
                    else:
                        n_per_row = n_reps
                    nn_patches = rearrange(nn_patches, 'b k h w c -> (b k) c h w').to(self.device)
                    out['retro_nns'] = nn_patches
                    out['retro_nns'] = make_grid(nn_patches, nrow=n_per_row, normalize=True)
                    out['retro_nns'] = 2. * out['retro_nns'] - 1.



        # TODO until now not working with conditional retrieval encoder (aka the one modulating the nns with the query)
        if self.conditional_retrieval_encoder:
            raise NotImplementedError('Conditional retrieval encoder not yet working')

        if isinstance(self.retrieval_encoder, (VQGANAggregator, VQGANNNAttender)):
            test_k = n_reps if n_reps is not None else k_nn
            c = self.retrieval_encoder(retro_cond, k=test_k)

        else:
            c = self.retrieval_encoder(retro_cond)
        print(c.shape)

        bs = c.shape[0]
        c_unconditional_guidance = self.get_unconditional_conditioning(c.shape,
                                                                        unconditional_guidance_label=unconditional_retro_guidance_label,
                                                                       k_nn=k_nn)
        if n_reps is not None:
            c_unconditional_guidance = torch.cat([c_unconditional_guidance]*n_reps,dim=1)

        if cond is not None:
            c = [c, cond]
            uncond_cond_cond = super().get_unconditional_conditioning(bs,
                                                                      unconditional_guidance_label=unconditional_guidance_label)

            c_unconditional_guidance = [c_unconditional_guidance, uncond_cond_cond]

        with self.ema_scope("Plotting"):
            samples, _ = self.sample_log(cond=c, batch_size=bs, unconditional_guidance_scale=unconditional_guidance_scale,
                                                      unconditional_conditioning=c_unconditional_guidance, **kwargs)

        x_samples = self.decode_first_stage(samples)
        out["query_samples"] = x_samples


        return out


    def get_qids(self,memsize,N,qids=None,use_weights=False,verbose=False):

        if isinstance(memsize,float) and hasattr(self,'nn_memory'):
            assert memsize > 0 and memsize <= 1., 'Require memsize in (0,1]'
            memsize = int(memsize*self.nn_memory.shape[0])

        if qids is None:
            if self.use_memory:
                memsize = min(memsize,self.nn_memory.shape[0])
                print(f'Top-M Sampling with memory size {memsize}')
                nn_mem = self.nn_memory.detach().cpu().numpy()[:memsize]

                if use_weights:
                    freqs = np.asarray([self.id_count[int(id_)] for id_ in nn_mem])
                    ps = freqs / freqs.sum(keepdims=True)
                else:
                    ps = None
                qids = np.random.choice(nn_mem, size=N,p=ps)
            else:
                print('Randomly sampling retrieval database entries')
                qids = np.random.choice(len(self.retriever.data_pool['embedding']), size=N)
        else:
            assert qids.shape[0] == N
        if verbose:
            print(f'Sampled entries are {qids}')



        return qids

    @torch.no_grad()
    def sample_from_rdata(self, N, cond=None,return_nns=False, use_weights=False,
                          qids=None, k_nn=None, memsize=100,verbose=False, pre_loaded_patches=None,
                          unconditional_guidance_scale=1.,unconditional_guidance_label=None,
                          unconditional_retro_guidance_label=None,
                          nn_embeddings=None,**kwargs):
        # only applicable when npatches = 1
        if self.retriever.searcher is None:
            self.train_searcher()

        if k_nn is None:
            k_nn = self.k_nn


        qids = self.get_qids(memsize, N, qids=qids,
                            use_weights=use_weights, verbose=verbose)

        out = {}

        try:
            query_embeddings = self.retriever.data_pool['embedding'][qids]

        except Exception as e:
            # for debug purposes
            print(f'Catchy: ', e)
            query_embeddings = torch.rand((N,512), dtype=torch.float).numpy()


        if pre_loaded_patches is None or self.nn_encoder is None:
            nns, distances = self.retriever.searcher.search_batched(
                query_embeddings / np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis]
                , final_num_neighbors=k_nn)
            patches = None
        else:
            patches = torch.stack([pre_loaded_patches[int(q)] for q in qids],0)
            if verbose:
                print(f'Using preloaded patches of shape {patches.shape}')


        qs = {}
        # TODO This assumes that we're only using one patch per query image
        in_shape = (N,1,k_nn)
        if self.nn_encoder is None:
            if nn_embeddings is None:
                retro_cond = torch.from_numpy(self.retriever.data_pool['embedding'][nns]).to(self.device).to(torch.float)
            else:
                retro_cond = nn_embeddings
            if return_nns:
                if patches is None:
                    patches = self.retriever.get_nn_patches(nns).to(torch.float)


                # out['batched_nns'] = torch.stack(batched_grids)
                out['batched_nns'] = rearrange(patches,'b n h w c -> b n c h w')

                patches = rearrange(patches, 'b n h w c -> (b n) c h w')
                img_patches = make_grid(patches, nrow=k_nn, normalize=True)
                out['sampled_nns'] = 2. * img_patches - 1.



        else:
            # TODO try oto change this for sampling. Maybe save nn_ids along with nn_info
            if patches is None:
                assert self.retriever.load_patch_dataset, 'Need to load patch dataset of retriever for obtaining nns as image patches'
                patches = self.retriever.get_nn_patches(nns).to(torch.float)
            patches = rearrange(patches, 'b n h w c -> (b n) c h w')
            img_patches = make_grid(patches, nrow=k_nn, normalize=True)
            out['sampled_nns'] = 2. * img_patches - 1.


            retro_cond = self.encode_with_fixed_nn_encoder(patches.to(self.device),in_shape)


        if self.conditional_retrieval_encoder:
            if self.nn_encoder is None:
                query_codes = rearrange(torch.from_numpy(query_embeddings).to(self.device).to(torch.float),
                                        'b c -> b 1 c')
            else:
                query_patches = self.retriever.get_nn_patches(qids[:, None])
                query_patches = rearrange(query_patches, 'b n h w c -> (b n) c h w')

                query_codes = self.encode_with_fixed_nn_encoder(query_patches,in_shape)


            qs = {'context': query_codes}

        c = self.retrieval_encoder(retro_cond, **qs)

        c_unconditional_guidance = self.get_unconditional_conditioning(c.shape,
                                                                       unconditional_guidance_label=unconditional_retro_guidance_label,
                                                                       k_nn=k_nn)

        if cond is not None:
            c = [c, cond]
            uncond_cond_cond = super().get_unconditional_conditioning(N,
                                                                      unconditional_guidance_label = unconditional_guidance_label)

            c_unconditional_guidance = [c_unconditional_guidance, uncond_cond_cond]

        with self.ema_scope("Plotting"):
            samples, z_denoise_row = self.sample_log(cond=c, batch_size=N,unconditional_guidance_scale=unconditional_guidance_scale,
                                                     unconditional_conditioning=c_unconditional_guidance,**kwargs)

        x_samples = self.decode_first_stage(samples)
        out["samples_with_sampled_nns"] = x_samples

        return out


    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,
                   custom_shape=None, del_sampler=False, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            if custom_shape is None:
                if hasattr(self,'sequence_length'):
                    shape = (self.channels,self.sequence_length, self.image_size, self.image_size)
                else:
                    shape = (self.channels, self.image_size, self.image_size)
            else:
                shape = custom_shape
            ddim_steps=kwargs.pop('S',ddim_steps)
            verbose = kwargs.pop('verbose',False)
            samples, intermediates = ddim_sampler.sample(S=ddim_steps, batch_size=batch_size,
                                                         shape=shape, conditioning=cond, verbose=verbose, **kwargs)
            if del_sampler:
                del ddim_sampler

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(
            self.retrieval_encoder.parameters())  # adding the encoder of retrieval embeddings/patches
        if self.conditional_retrieval_encoder:
            params += list(self.query_encoder.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_nn_encoder:
            print(f"{self.__class__.__name__}: Also optimizing nn_encoder params!")
            params = params + list(self.nn_encoder.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
