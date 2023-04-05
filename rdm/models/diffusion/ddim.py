"""SAMPLING ONLY."""
from functools import partial

import kornia
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm

from ldm.modules.diffusionmodules.util import (make_ddim_sampling_parameters,
                                               make_ddim_timesteps)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,   # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               random_guiding='none',
               r_shape=None,
               retro_cond=None,
               return_neighbors=False,
               k_nn=None,
               ignore_noising=False,
               content_cond=None,
               style_cond=None,
               intermediates_to_cpu=False,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            if isinstance(conditioning, list):
                for c in conditioning:
                    if c.shape[0] != batch_size:
                        print(f"Warning: Got {c.shape[0]} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

            if unconditional_guidance_scale > 1.:
                print(f'Using unconditonal diffusion guidance with scale {unconditional_guidance_scale}')

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        assert random_guiding in ['none','sampled','const'], f'Unknown random guidance option {random_guiding}'
        # sampling
        # C, H, W = shape
        # size = (batch_size, C, H, W)
        size = (batch_size,) + tuple(shape)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(cond=conditioning,
                                                    shape=size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    r_shape=r_shape,
                                                    retro_cond=retro_cond,
                                                    return_neighbors=return_neighbors,
                                                    k_nn=k_nn,
                                                    ignore_noising=ignore_noising,
                                                    random_guiding=random_guiding,
                                                    content_cond=content_cond,
                                                    style_cond=style_cond,
                                                    intermediates_to_cpu=intermediates_to_cpu
                                                    )
        return samples.detach(), intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      random_guiding='none',content_cond=None, style_cond=None, intermediates_to_cpu=False, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        random_guider = None
        if random_guiding != 'none':
            print(f'Sample with random guiding of type {random_guiding}')
            random_guider = torch.clamp(torch.randn(shape,device=self.model.device),-1.,1.)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            snr = self.ddim_alphas[index] / (1 - self.ddim_alphas[index])

            # for disseecting style (color etc.) and content as in https://arxiv.org/abs/2204.00227
            input_cond = cond
            if style_cond is not None and snr < 5.e-2:
                input_cond = style_cond
            if content_cond is not None and snr >= 5.e-2 and snr < 1.:
                input_cond = content_cond

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if random_guiding == 'sampled':
                random_guider = torch.clamp(torch.randn(shape,device=self.model.device),-1.,1.)

            outs = self.p_sample_ddim(img, input_cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      random_guider=random_guider,
                                      )
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                if intermediates_to_cpu:
                    intermediates['x_inter'].append(img.detach().cpu())
                    intermediates['pred_x0'].append(pred_x0.detach().cpu())
                else:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,noise=None,
                      random_guider=None):
        b, *_, device = *x.shape, x.device
        assert unconditional_guidance_scale >= 1.


        if noise is None:
            noise = torch.randn(x.shape, device=device)

        if unconditional_guidance_scale > 1.:
            # if random_guider == None:
            assert unconditional_conditioning is not None
            combined_c = torch.cat([c,unconditional_conditioning],dim=0)
            combined_x = torch.cat([x]*2,dim=0)
            combined_t = torch.cat([t]*2,dim=0)
            combined_out =  self.model.apply_model(combined_x, combined_t, combined_c)
            e_t = combined_out[:b]
            e_t_uncond = combined_out[b:]
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            # else:
            #     e_t = random_guider + unconditional_guidance_scale *(e_t - random_guider)
        else:
            e_t = self.model.apply_model(x, t, c)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full_like(e_t, alphas[index])
        a_prev = torch.full_like(e_t, alphas_prev[index])
        sigma_t = torch.full_like(e_t, sigmas[index])
        sqrt_one_minus_at = torch.full_like(e_t, sqrt_one_minus_alphas[index])

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

class DDIMRetroSampler(DDIMSampler):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from ldm.models.diffusion.ddpm import PreNoiserRetroDiffusion
        assert isinstance(self.model,PreNoiserRetroDiffusion), 'Model needs to be of type MinimalRETRODiffusion'

    @torch.no_grad()
    def ddim_sampling(self, cond, retro_cond, shape, r_shape=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      return_neighbors=False,k_nn=None, ignore_noising=False,
                      ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if r_shape is None:
            r_shape = shape

        if retro_cond is None:
            rc = torch.randn(r_shape, device=device)

            if self.model.conditional_retrieval_encoder:
                qs = {'context': img}
            else:
                qs = {}

            # run retrieval encoder to obtain shape of conditioning
            r_enc = self.model.retrieval_encoder(rc, **qs)

        else:
            # if self.model.pre_noise:
            r_enc = retro_cond
            r_enc = self.model.retrieval_encoder(r_enc)



        if not self.model.pre_noise:
            # noise is added after encoding data with retrieval encoder
            r_enc = torch.randn_like(r_enc) if retro_cond is None else retro_cond


        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [], 'pred_x0': [],}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if cond is None:
                c_cond = [r_enc]
            else:
                c_cond = [r_enc,cond]

            noise = torch.randn(shape,device=device)
            outs = self.p_sample_ddim(img, c_cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      noise=noise)
            img, pred_x0 = outs
            if retro_cond is None:
                # get retrieval emebedding based on x0_pred
                px0 = self.model.decode_first_stage(pred_x0)
                retro_conditionings = self.model.get_nn_and_encoding(px0,
                                                                     return_patches=return_neighbors,
                                                                     return_query_patches=return_neighbors, # also return query patches when returning neighbors
                                                                     k_nn=k_nn)
                rc = retro_conditionings[self.model.nn_key]
                if return_neighbors:
                    query_patches = retro_conditionings['query_patches']  # dim = [b, n_ptch, 1, c, h, w ]
                    nn_patches = kornia.geometry.resize(rearrange(retro_conditionings['image_patches'],'b (n k) h w c -> (b n k) c h w',
                                                                  n=self.model.n_patches_per_side ** 2, k=k_nn if k_nn is not None else self.model.k_nn,
                                                                  b=b), size=query_patches.shape[-1])
                    nn_patches = rearrange(nn_patches,'(b n k) c h w -> b n k c h w',
                                           n=self.model.n_patches_per_side**2,k=k_nn if k_nn is not None else self.model.k_nn,
                                           b=b).type_as(rc) # dim = [b, n_ptch, k, h, w, c ]

                    nn_patches = rearrange(torch.cat([query_patches,nn_patches],dim=2), 'b n k c h w -> (b n k) c h w')

                if rc.ndim==4:
                    rc = rearrange(rc, 'b n k d -> b (n k) d')
                if self.model.pre_noise:
                    # rescale and noise: since nn_embedings are within [0,1] this is possible
                    rc = rc * 2. - 1.
                    if not ignore_noising:
                        rc = self.model.q_sample(rc, ts)
            else:
                # get retrieval embedding from retro cond from database and noise (as done during training)
                if self.model.pre_noise:
                    rc = retro_cond * 2. - 1.
                    if not ignore_noising:
                        rc = self.model.q_sample(rc,ts)
                else:
                    rc = retro_cond


            if self.model.conditional_retrieval_encoder:
                zq = self.model.q_sample(pred_x0,ts,noise=noise)
                qs = {'context': self.model.query_encoder(zq)}
            else:
                qs = {}

            r_enc = self.model.retrieval_encoder(rc, **qs)  # can also be initialized to the identity

            if not self.model.pre_noise:
                # clamp output to be within [-1,1]
                r_enc = self.model.adjust_support(r_enc)
                if not ignore_noising:
                    r_enc = self.model.q_sample(r_enc, ts)


            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                if return_neighbors:
                    intermediates[f'intermediate_retro_nns@{index}'] = nn_patches

        return img, intermediates
