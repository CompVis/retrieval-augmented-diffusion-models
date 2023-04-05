import glob
import os
import shutil
import tarfile

import albumentations
import cv2
import numpy as np
import taming.data.utils as tdu
import yaml
from omegaconf import OmegaConf
from PIL import Image
from taming.data.imagenet import (download, give_synsets_from_indices,
                                  retrieve, str_to_indices)
from torch.utils.data import Dataset
from tqdm import tqdm

from rdm.data.base import ImagePaths, PRNGMixin


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())


class ImageNetBase(Dataset, PRNGMixin):
    def __init__(self, config=None, p_unconditional=0.):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()
        self.p_unconditional = p_unconditional
        if self.p_unconditional > 0.:
            print(f"{self.__class__.__name__}: Replacing class labels with extra label '1000' with a probability "
                  f"p={p_unconditional:.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        if self.p_unconditional > 0.:
            if self.prng.random() <= self.p_unconditional:
                # classes in AFHQ run from 0 to 2
                example["class_label"] = 1000
                example["human_label"] = "unknown"
        return example

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def get_subset_by_label_range(self,
                                 label_key:str,
                                 label_range:list,
                                 **kwargs):
        if len(label_range) == 2:
            label_range = np.arange(label_range[0],
                                    label_range[1]+1)
        return self.data.labels['id'][np.isin(self.data.labels[label_key],label_range)]


    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)
            self.human2integer_dict["unknown"] = 1000

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
            "id": np.arange(len(self.relpaths))
        }

        self._load_embedding_file()
        if hasattr(self, "img_embeddings"):
            labels["clip_img_emb"] = self.img_embeddings["embedding"]

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths

    def _load_embedding_file(self):
        saved_embeddings = self.config.get("clip_img_embeddings", None)
        if saved_embeddings:
            print(f"{self.__class__.__name__}: loading custom_clip image embeddings from {saved_embeddings}")
            assert saved_embeddings.endswith('.npz'), 'saved embeddings not stored as a .npz file'
            compressed = np.load(saved_embeddings)
            data_pool = {key: compressed[key] for key in compressed.files}
            assert data_pool['embedding'].shape[0] == len(self.abspaths), 'do not have an embedding for every example'
            print('Finished loading of patch embeddings.')
            self.img_embeddings = data_pool


class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)



class BigGANImageNet(Dataset):
    def __init__(self, size, lr_size=None, zoom_pixels=None):
        self.base = self.init_base() # unscaled images
        self.size = size
        self.lr_size = lr_size
        self.zoom_pixels = zoom_pixels  # central crop, works as conditioning and also resized to size
        if zoom_pixels is not None: assert self.zoom_pixels > 0
    def __len__(self):
        return len(self.base)

    def preprocess(self, x):
        dtype = x.dtype
        assert dtype in [np.float32, np.float64]
        assert x.min() >= -1
        assert x.max() <= 1
        x = (x+1.0)*127.5
        x = x.clip(0,255).astype(np.uint8)

        h, w, _ = x.shape
        # largest central crop
        b = min(h, w)
        h0 = (h-b)//2
        w0 = (w-b)//2
        tmp = x[h0:h0+b,w0:w0+b,:]

        zoom = None
        # optional zoom
        if self.zoom_pixels is not None:
            hz, wz, _ = tmp.shape
            bz = self.zoom_pixels // 2
            zoom = tmp[bz:hz-bz, bz:wz-bz, :]
            zoom = Image.fromarray(zoom)
            zoom = zoom.resize((self.size, self.size), Image.BILINEAR)
            zoom = np.array(zoom)/127.5-1.0
            zoom = zoom.astype(dtype)

        x = tmp
        # pillow bilinear instead of tensorflow area resizing
        x = Image.fromarray(x)
        x = x.resize((self.size, self.size), Image.BILINEAR)

        # optional lr image
        if self.lr_size is not None:
            lr = x.resize((self.lr_size, self.lr_size), Image.BICUBIC)
            lr = lr.resize((self.size, self.size), Image.BICUBIC)
            lr = np.array(lr)/127.5-1.0
            lr = lr.astype(dtype)
        else:
            lr = None
        x = np.array(x)/127.5-1.0
        x = x.astype(dtype)

        return x, lr, zoom

    def __getitem__(self, i):
        example = self.base[i]
        x, lr, zoom = self.preprocess(example["image"])
        example["image"] = x
        if zoom is not None:
            example["image_zoom"] = zoom
        if self.lr_size is not None:
            example["lr"] = lr
        return example


class BigGANImageNetTrain(BigGANImageNet):
    def init_base(self):
        return ImageNetTrain()


class BigGANImageNetValidation(BigGANImageNet):
    def init_base(self):
        return ImageNetValidation()


if __name__ == "__main__":
    import torch.multiprocessing
    from einops import rearrange
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from rdm.modules.retrievers import ClipImageRetriever
    torch.multiprocessing.set_sharing_strategy('file_system')

    def get_input(batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    # saves CLIP embeddings of train data
    embedder = ClipImageRetriever(model="ViT-B/32").cuda()
    dset = ImageNetTrain(config={"size": 256})
    dloader = DataLoader(dset, batch_size=200, drop_last=False, shuffle=False, num_workers=20)

    savepath = f"data/clip_img_embeddings/{dset.__class__.__name__}"
    identifier = f"{dset.__class__.__name__}-256x256-embeddings-and-ids"

    # TODO: for datasets up to 12M examples, we can simply store a large array in memory and save at once (~1GB RAM per 1M examples => ~12GB for CC12M, ...)
    # todo: For larger datasets, however, have to rely on chunked saving/reading or some other file format like hdf5
    out = {"emb": [], "id": []}
    for batch in tqdm(dloader, desc="Saving Embeddings"):
        x = get_input(batch, "image").cuda()
        emb = embedder(x)
        # save to disk, image id and corresponding embedding
        emb = emb.detach().cpu().numpy()
        out["emb"].append(emb)
        out["id"].append(batch["id"])

    out["emb"] = np.concatenate(out["emb"])
    out["id"] = np.concatenate(out["id"])

    print("saving...")
    print(f"output embeddings have shape {out['emb'].shape}")
    print(f"output ids have shape {out['id'].shape}")

    saved_embeddings = f'{savepath}/{identifier}.npz'
    os.makedirs(savepath, exist_ok=True)

    np.savez_compressed(saved_embeddings,
                        embedding=out["emb"],
                        id=out["id"]
        )
    print("done.")
