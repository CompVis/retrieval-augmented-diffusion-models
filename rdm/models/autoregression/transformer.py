import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config, log_txt_as_img
from taming.models.cond_transformer import Net2NetTransformer

from rdm.modules.encoders.nn_encoders import IdentityEncoder


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentCrossTransformer(Net2NetTransformer):
    def __init__(self, scheduler_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pkeep == 1.0, 'currently only supporting pkeep=1.0'
        sos = self.sos_token
        del self.sos_token
        self.register_buffer("sos_token", torch.LongTensor([sos]))
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

    def forward(self, x, c, e=None):
        assert e is None, 'support currently dropped'
        _, z_indices = self.encode_to_z(x)
        _, cond = self.encode_to_c(c)
        target = z_indices
        sos = repeat(self.sos_token, '... -> b (...)', b=x.shape[0])
        z_indices = torch.cat((sos, z_indices), 1)[:, :-1]
        logits, _ = self.transformer(cond, z_indices)
        return logits, target

    def compute_loss(self, logits, targets, split="train"):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, {f"{split}/loss": loss.detach()}

    def training_step(self, batch, batch_idx):
        logits, target = self.shared_step(batch, batch_idx)
        loss, log_dict = self.compute_loss(logits, target, split="train")
        self.log("train/loss", loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        e = self.get_e(batch)
        logits, target = self(x, c, e=e)
        return logits, target

    def validation_step(self, batch, batch_idx):
        logits, target = self.shared_step(batch, batch_idx)
        loss, log_dict = self.compute_loss(logits, target, split="val")
        self.log("val/loss", loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return log_dict

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        assert not self.transformer.training
        enc = self.transformer.encoder(c, mask=None, return_embeddings=True)
        sos = repeat(self.sos_token, '... -> b (...)', b=x.shape[0])
        x = torch.cat((sos, x), 1)
        for k in range(steps):
            callback(k)
            # logits = self.transformer.decoder(x, context=enc, mask=None, context_mask=None)
            logits, _ = self.transformer(c, x, enc=enc)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        x = x[:, 1:]  # cut of the sos token
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [optimizer], scheduler
        return optimizer


class LatentImageRETRO(LatentCrossTransformer):
    """
    predict tokens in VQ Space:
     - conditioning on anything other than RETRO tokens happens via concatenation (i.e. the seq-len is len(c)+len(z))
     - conditioning on RETRO tokens via cross-attention
     - expect that retro-conditioning comes from dataloader (modulo optional nn-encoder)
     - the encoder can also be conditioned on the query (via cross-attention)
    TODO:
        - add query conditioning
        - implement conditioning via cross-attention on c (needs a dedicated class/pl-module)
        - chunked cross attention?
        - "causal" masking as in the retro-paper? currently conditioning on the whole chunk
    """
    def __init__(self, nn_encoder_cfg, nn_key, mask_token, p_mask_max=0., nn_reshaper_cfg=None, nn_memory=None, retrieval_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask_token", torch.LongTensor([mask_token]))  # TODO: also need to handle vectors...
        self.p_mask_max = p_mask_max
        self.nn_key = nn_key
        self.nn_encoder = instantiate_from_config(nn_encoder_cfg).eval()
        self.nn_encoder.train = disabled_train
        if nn_reshaper_cfg is not None:
            self.nn_reshaper = instantiate_from_config(nn_reshaper_cfg)
        else:
            self.nn_reshaper = torch.nn.Identity()

        self.use_memory = nn_memory is not None
        if self.use_memory:
            assert os.path.isfile(nn_memory) and nn_memory.endswith('.p')
            with open(nn_memory,'rb') as f:
                nn_data = pickle.load(f)
            self.register_buffer('nn_memory',torch.tensor(nn_data['nn_memory'],dtype=torch.int),persistent=False)
            print(f'Loaded nn_memory of size {self.nn_memory.shape[0]}')
            self.id_count = nn_data['id_count']

        for param in self.nn_encoder.parameters():
            param.requires_grad = False

        self.retriever = None
        self.init_retriever(retrieval_cfg)

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

    @rank_zero_only
    def train_searcher(self):
        print("training searcher...")
        self.retriever.train_searcher()
        print("done training searcher")

    @torch.no_grad()
    def encode_nns(self, nns):
        # identity in case of vq-indices or clip-embeddings, but can also be vq-first stage etc
        # nn-encoder handles the logic of reshaping to b, seqlen, dim (e.g. seqlen = n_patch * k for concat strategy)
        nn_encodings = self.nn_encoder.encode(nns)
        return nn_encodings

    def get_mask_prob(self):
        # TODO: implement more fancy stuff like skewed distributions etc
        p = np.random.uniform(0., self.p_mask_max)
        return p

    @torch.no_grad()
    def get_r(self, batch, N=None, p_mask=0.):
        # get them retro conditionings
        nns = batch[self.nn_key]
        if N is not None:
            nns = nns[:N]
        nns = self.nn_reshaper(nns)
        r = self.encode_nns(nns)   # shape: b, s, d  (e.g. s = n_patch * k for concat strategy)

        if p_mask > 0.:
            # apply masking on retro-conditoning
            mask = torch.bernoulli(torch.ones_like(r) * p_mask)
            mask = mask.round().to(dtype=torch.int64)
            r = r * (1-mask) + mask * torch.ones_like(r) * self.mask_token
        return r

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        r = self.get_r(batch, p_mask=self.get_mask_prob())
        logits, target = self(x, c, r)
        return logits, target

    def forward(self, x, c, r):
        # r is the retro-conditioning (i.e. nns)
        # we expect they are already preprocessed such that the can be used by the transformer
        # condition the model via cross-attn on r, and also via concat/additional cross-attn on other modalities
        _, z_indices = self.encode_to_z(x)  # image tokens
        _, cond = self.encode_to_c(c)  # e.g. text
        target = z_indices
        z_indices = torch.cat((cond, z_indices), 1)[:, :-1]
        logits = self.transformer(z_indices, context=r)  # TODO: add query conditioning  --> adopt configure_optimizers
        return logits[:, cond.shape[1]-1:], target

    @torch.no_grad()
    def sample(self, x, r, c, steps, temperature=1.0, sample=False, top_k=None, guidance_scale=1.0,
               callback=lambda k: None, **kwargs):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        assert not self.transformer.training
        x = torch.cat((c, x), 1)

        bs = x.shape[0]
        if guidance_scale > 1.0:
            empty_r = torch.zeros_like(r)
            r = torch.cat((r, empty_r), dim=0)

        for k in range(steps):
            callback(k)
            # logits = self.transformer.decoder(x, context=enc, mask=None, context_mask=None)
            # logits = self.transformer(x, context=r)

            if guidance_scale > 1.0:
                x = torch.cat((x, x), dim=0)
            logits = self.transformer(x, context=r)
            if guidance_scale > 1.0:
                x = x[:bs]
                logits_cond = logits[:bs]
                logits_uncond = logits[bs:]
                logits = logits_uncond + guidance_scale*(logits_cond - logits_uncond)

            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        x = x[:, c.shape[1]:]  # cut of the conditioning tokens
        return x

    @torch.no_grad()
    def decode_from_precode(self, prequant, h, w, c):
        quant_z = rearrange(prequant, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def sampling_util(self, steps, z_start, r, c, temperature, top_k, zshape, callback=None, top_p=1., **kwargs):
        assert top_p==1., 'not yet implemented'   # TODO
        t1 = time.time()
        index_sample = self.sample(z_start, r, c,
                                   steps=steps,
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None,
                                   **kwargs,
                                   )
        if not hasattr(self, "sampling_time"):
            self.sampling_time = time.time() - t1
            print(f"Full sampling takes about {self.sampling_time:.2f} seconds.")
        x_sample = self.decode_to_img(index_sample, zshape)
        return x_sample

    @torch.no_grad()
    def sample_from_rdata(self, N, cond=None,return_nns=False, use_weights=False,
                          qids=None, k_nn=None, memsize=100,verbose=False,
                          top_k=256, temperature=1.0,
                          code_side_len=16, z_dimensionality=256, # information about the first stage model
                          pre_loaded_patches=None, nn_embeddings=None, query_embeddings=None,
                          **kwargs):
        # only applicable when npatches = 1
        if self.retriever.searcher is None:
            self.train_searcher()

        if k_nn is None:
            k_nn = self.k_nn

        out = {}

        nns = None
        if nn_embeddings is None:
            if query_embeddings is None:
                qids = self.get_qids(memsize, N, qids=qids,
                                    use_weights=use_weights, verbose=verbose)
                out["qids"] = qids

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
        if self.nn_encoder is None or isinstance(self.nn_encoder, IdentityEncoder):
            if nn_embeddings is None:
                retro_cond = torch.from_numpy(self.retriever.data_pool['embedding'][nns]).to(self.device).to(torch.float)
            else:
                retro_cond = nn_embeddings

            if return_nns and nns is not None:
                if patches is None:
                    patches = self.retriever.get_nn_patches(nns).to(torch.float)


                # out['batched_nns'] = torch.stack(batched_grids)
                out['batched_nns'] = rearrange(patches,'b n h w c -> b n c h w')

                patches = rearrange(patches, 'b n h w c -> (b n) c h w')
                img_patches = make_grid(patches, nrow=k_nn, normalize=True).unsqueeze(0)
                out['sampled_nns'] = 2. * img_patches - 1.


                if qids is not None:
                    q_patches = self.retriever.get_nn_patches(torch.tensor(qids)[:, None]).to(torch.float)
                    q_patches = rearrange(q_patches,'b n h w c -> (b n) c h w')
                    out['batched_queries'] = q_patches

                    # img_patches = make_grid(q_patches, nrow=1, normalize=True)
                    out['sampled_queries'] = q_patches
        else:
            # TODO try oto change this for sampling. Maybe save nn_ids along with nn_info
            if patches is None:
                assert self.retriever.load_patch_dataset, 'Need to load patch dataset of retriever for obtaining nns as image patches'
                patches = self.retriever.get_nn_patches(nns).to(torch.float)
            patches = rearrange(patches, 'b n h w c -> (b n) c h w')
            img_patches = make_grid(patches, nrow=k_nn, normalize=True)
            out['sampled_nns'] = 2. * img_patches - 1.


            retro_cond = self.encode_with_fixed_nn_encoder(patches.to(self.device),in_shape)

        if cond is None:
            # assumes SOSProvider as conditioning
            _, cond = self.encode_to_c(torch.zeros((N, 0)))
            cond = cond.to(retro_cond.device)
        else:
            raise NotImplementedError()

        z_shape = (N, z_dimensionality, code_side_len, code_side_len)
        steps = code_side_len**2
        z_start = torch.zeros((N, 0), device=retro_cond.device, dtype=torch.long)
        samples = self.sampling_util(steps, z_start, retro_cond, cond, temperature, top_k, z_shape, **kwargs)
        out["samples_with_sampled_nns"] = samples

        return out


    def get_qids(self,memsize,N,qids=None,use_weights=False,verbose=False):
        if isinstance(memsize,float):
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
    def log_images(self,
                   batch,
                   temperature=None,
                   top_k=256,
                   top_p=1.0,
                   callback=None,
                   N=4,
                   half_sample=True,
                   sample=True,
                   p_sample=True,
                   plot_cond_stage=True,
                   patch_plotter_cfg=None,
                   masking_probs=[0.5, 1.0],
                   memsize=1.0,
                   **kwargs
                   ):
        log = dict()
        x, c = self.get_xc(batch, N)
        r = self.get_r(batch, N, p_mask=0.)
        x = x.to(device=self.device)
        r = r.to(device=self.device)

        if type(c) != list:
            c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        if sample:
            z_start_indices = z_indices[:, :0]
            x_sample = self.sampling_util(z_indices.shape[1], z_start_indices, r, c_indices, zshape=quant_z.shape,
                                          temperature=temperature, top_k=top_k, top_p=top_p, callback=callback)
            log["samples_full"] = x_sample

        if half_sample:
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            x_sample = self.sampling_util(z_indices.shape[1]-z_start_indices.shape[1], z_start_indices, r, c_indices,
                                          temperature=temperature, top_k=top_k, top_p=top_p, callback=callback,
                                          zshape=quant_z.shape)
            log["samples_half"] = x_sample

        if p_sample:
            if masking_probs[0] >= self.p_mask_max and self.p_mask_max != 0.:
                masking_probs = [self.p_mask_max] + masking_probs
            for p_mask in masking_probs:
                r = self.get_r(batch, N, p_mask=p_mask)   # TODO: check again with the masking
                r = r.to(device=self.device)
                z_start_indices = z_indices[:, :0]
                x_sample = self.sampling_util(z_indices.shape[1], z_start_indices, r, c_indices, zshape=quant_z.shape,
                                              temperature=temperature, top_k=top_k, top_p=top_p, callback=callback)
                log[f"samples_full_p_{p_mask:.2f}"] = x_sample

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if plot_cond_stage:
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec

            if self.cond_stage_key == "caption":
                del log["conditioning_rec"]
                c = log_txt_as_img((x.shape[2], x.shape[3]), c)
            log["conditioning"] = c

        if self.retriever is not None:
            additional = self.sample_from_rdata(
                z_indices.shape[0], return_nns=True,k_nn=r.shape[1],
                top_k=top_k,temperature=temperature, memsize=memsize,
                code_side_len=quant_z.shape[-1], z_dimensionality=quant_z.shape[1]
            )
            log["_samples_with_sampled_nns"] = additional["samples_with_sampled_nns"]
            log["_sampled_nns"]              = additional["sampled_nns"]
            log["_sampled_queries"]          = additional["sampled_queries"]

        if patch_plotter_cfg is not None:
            patch_plotter = instantiate_from_config(patch_plotter_cfg)
            grid = patch_plotter(batch, N)
            log["neighbors"] = grid
            del patch_plotter

        return log


class NNReshaper(object):
    def __call__(self, x):
        nn_patches = rearrange(x, 'b n k h w c -> (b n k) c h w')#.to(torch.float)
        return nn_patches


class NNEmbeddingReshaper(object):
    def __call__(self, x):
        nn_patches = rearrange(x, 'b n k d -> b (n k) d')#.to(torch.float)
        return nn_patches


class ImageNeighborPlotter(object):
    """Directly retrieving images, makes a grid"""
    def __init__(self, nn_key):
        self.nn_key = nn_key

    def __call__(self, batch, N=None):
        x = batch[self.nn_key]
        if N is not None:
            x = x[:N]
        k = x.shape[2]
        nn_patches = rearrange(x, 'b n k h w c -> (b n k) c h w')  # .to(torch.float)
        grid = make_grid(nn_patches, nrow=k, normalize=True)
        grid = 2. * grid - 1.
        return grid
