import random
from functools import partial

import numpy as np
import torch
from einops import rearrange

from ldm.util import get_obj_from_str


def isvideo(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (x.ndim == 5) and (x.shape[1] == 3 or x.shape[1] == 1)


def ischannellastimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[-1] == 3 or x.shape[-1] == 1)


def convert_nn_tree(nns):
    for n_ptch in nns:

        for key in nns[n_ptch]:
            if isinstance(nns[n_ptch][key],np.ndarray) and nns[n_ptch][key].dtype == np.uint32:
                nns[n_ptch][key] = nns[n_ptch][key].astype(np.int32)

    return nns


def load_partial_from_config(config):
    return partial(get_obj_from_str(config['target']),**config.get('params',dict()))


def crop_coords(img_size, crop_size,random_crop:bool):
    assert crop_size<=min(img_size)
    height, width = img_size
    if random_crop:
        # random crop
        h_start = random.random()
        w_start = random.random()
        y1 = int((height - crop_size) * h_start)
        x1 = int((width - crop_size) * w_start)
    else:
        # center crop
        y1 = (height - crop_size) // 2
        x1 = (width - crop_size) // 2

    return x1, y1


def make_video_grid(vid_batch, nrow=4):
    vid_batch = vid_batch.detach().cpu()
    pad = vid_batch.shape[0] % nrow
    if pad != 0:
        vid_batch = torch.cat([vid_batch]+[torch.full_like(vid_batch[0][None],0)]*pad)

    rows = []
    for r in range(vid_batch.shape[0] // nrow):
        row = vid_batch[r*nrow:(r+1)*nrow]
        rows.append(torch.cat(list(row),dim=-1))

    grid = torch.cat(rows,dim=-2)
    grid = rearrange(grid, 'c t h w -> t h w c')
    return grid
