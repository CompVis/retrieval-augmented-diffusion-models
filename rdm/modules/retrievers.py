import kornia
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from clip import load as load_clip

from main import instantiate_from_config
from rdm.modules.custom_clip.clip import tokenize as custom_tokenize


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VQGANRetriever(nn.Module):
    def __init__(self, embedder_config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device=device
        self.model = self.get_retriever_model(embedder_config, device)

    def get_retriever_model(self, config, device):
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        model = model.eval()
        model.train = disabled_train
        return model.to(device)

    @torch.no_grad()
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (256, 256), interpolation='bicubic',align_corners=True)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        x = x.to(self.device)
        return self.model.encode(self.preprocess(x)).sample().reshape(x.shape[0], -1)


class VAERetriever(nn.Module):
    def __init__(self, embedder_config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device=device
        self.model = self.get_retriever_model(embedder_config, device)

    def get_retriever_model(self, config, device):
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        model = model.eval()
        model.train = disabled_train
        return model.to(device)

    @torch.no_grad()
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (256, 256), interpolation='bicubic',align_corners=True)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        x = x.to(self.device)
        # TODO do we want to sample from the posterior?
        return self.model.encode(self.preprocess(x)).sample().reshape(x.shape[0], -1)


class ClipImageRetriever(nn.Module):
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = load_clip(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


class CLIPTextEmbedder(nn.Module):
    """
    return the text embedding given a batch of text prompts
    """
    def __init__(self, model="ViT-B/32", device="cuda",
                 add_k_shape=False):
        super(CLIPTextEmbedder, self).__init__()
        model, _ = load_clip(model, device=device)
        self.model = model
        self.device = device
        self.add_k_shape = add_k_shape

    def preprocess(self, text):
        return custom_tokenize(text)

    def forward(self, txt):
        emb = self.model.encode_text(self.preprocess(txt).to(self.device))
        if self.add_k_shape:
            emb = emb[:,None]
        return emb


class ClipTxt2ImageRetriever(CLIPTextEmbedder):
    """
    alias for. CLIPTextEmbedder
    return the text embedding given a batch of text prompts
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CLIPCutterTextEmbedder(nn.Module):
    """
    return the text embedding given a batch of text prompts
    """
    def __init__(self, model="ViT-B/32", device="cuda",jit=False):
        super().__init__()

        model, _ = load_clip(model, device=device,jit=jit)
        self.model = model
        self.device = device

    def preprocess(self, text):
        return custom_tokenize(text)

    def forward(self, txt):
        emb = self.model.encode_text(self.preprocess(txt).to(self.device))
        return emb
