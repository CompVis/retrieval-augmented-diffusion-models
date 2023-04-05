import numpy as np
import torch


def prepare_cond_rep(rep):
    return rep[None]


def get_k_nearest_from_embeddings(emb,knn):
    if emb.shape[0]>=knn:
        return emb[:knn][None]
    else:
        n_reps = knn // emb.shape[0]
        emb = np.concatenate([emb]*n_reps)
        remaining = emb[:(knn - emb.shape[0])]
        if len(remaining.shape) == 1:
            remaining=remaining[None]
        emb = np.concatenate([emb,remaining])

        return emb[None]

def reconstruct_nns(nn_ids,knn,index, sample_range):
    # remove faulty ids
    nn_ids = nn_ids[nn_ids!=-1]
    if nn_ids.size==0:
        # fallback
        print('fallback as no neighbors found')
        embds = np.zeros((1,knn,768))
    else:
        # sample to avoid duplicates and increase generalization
        nn_ids = np.random.choice(min(len(nn_ids),sample_range),
                                  size=knn,replace=False)

        embds = []
        for idx in nn_ids:
            rec_embds = index.reconstruct(int(idx))
            embds.append(rec_embds)
        # add extra dimension to account for n_pathces which is here always 1
        embds = np.stack(embds)[None]

    return embds


def extract_nns(nns, knn, n_patches=1):
    nns = nns[n_patches]
    return nns['embeddings'][:, :knn]

def load_txt(data):
    return data.decode('utf-8')

def load_int(data):
    return int(data)

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True,
                      ignore_keys=None):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched if not (ignore_keys is not None and key in ignore_keys)]


    result = {}
    for key in batched:
        if ignore_keys and key in ignore_keys:
            continue
        try:
            if isinstance(batched[key][0], (int, float)):
                if combine_scalars:
                    result[key] = np.array(list(batched[key]))
            elif isinstance(batched[key][0], torch.Tensor):
                if combine_tensors:
                    # import torch

                    result[key] = torch.stack(list(batched[key]))
            elif isinstance(batched[key][0], np.ndarray):
                if combine_tensors:
                    result[key] = np.array(list(batched[key]))
            else:
                result[key] = list(batched[key])
        except Exception as e:
            print(key)
            raise e
        # result.append(b)
    return result


