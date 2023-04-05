import ctypes
import os
import pickle
import sys
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, sharedctypes

import cv2
import numpy as np
import torch
from einops import rearrange
from kornia.geometry.transform import crop_by_boxes
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm

from ldm.util import instantiate_from_config, parallel_data_prefetch

from rdm.data.retrieval_dataset.dsetbuilder import DatasetBuilder


rescale = lambda x: (x + 1.) / 2.

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '-rc',
        '--rconfig',
        required=True,
        type=str,
        help='Config for the retrieval databuilder'
    )
    parser.add_argument(
        '-qc',
        '--qconfig',
        required=True,
        type=str,
        help='Config for the query dataset'
    )
    parser.add_argument(
        '-s',
        '--split',
        default='train',
        choices=['train', 'validation', 'test'],
        help='Split for the query dataset?'
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        default=160,
        type=int,
        help='Batch size for batched search'
    )
    parser.add_argument(
        '-nns',
        '--nns_savedir',
        default='neighbors',
        type=str,
        help='Savedir for the precomputed nns.'
    )
    parser.add_argument(
        '-lm',
        '--log_max_np',
        default=0,
        type=int,
        help='log2 of maximum number of patches per side.'
    )
    parser.add_argument(
        '-w',
        '--n_workers',
        default=50,
        type=int,
        help='Number of workers for dataloading.'
    )
    parser.add_argument(
        '-p',
        '--only_patches',
        default=False,
        action='store_true',
        help='only load patches?'
    )
    parser.add_argument(
        '-n',
        '--only_neighbors',
        default=False,
        action='store_true',
        help='only compute nns?'
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        choices=['text','img'],
        default='img',
        help='Similarity mode (for the retriever) when computing nearest neighbours.'
    )
    parser.add_argument(
        '-ps',
        '--parts',
        type=int,
        default=1,
        help='Number of parts to divide the overall dataset into'
    )
    parser.add_argument(
        '-pn',
        '--part_no',
        type=int,
        default=1,
        help='The part number for the actual run, only used if --parts>1'
    )
    return parser


class PatchSaverDataset(Dataset):
    def __init__(self, data: Dataset, filepath, n_patches_per_side, k, q_info):
        super().__init__()
        self.data = data
        self.basepath = '/'.join(filepath.split('/')[:-1])
        self.n_patches_per_side = n_patches_per_side
        self.n_patches = self.n_patches_per_side**2
        with open(filepath, 'rb') as f:
            self.nn_paths = pickle.load(f)

        if os.path.isfile(os.path.join(self.basepath,'corrupts.txt')):
            with open(os.path.join(self.basepath,'corrupts.txt'),'r') as f:
                self.corrupts = [line.rstrip() for line in f]
        else:
            self.corrupts = []

        # this is the info ordered after the img ids of the retrieval dataset
        self.q_info = q_info
        self.s_len = 50

        # hacky solution to work around PEP 3118 issue see https://stackoverflow.com/questions/45693949/storing-strings-in-a-multiprocessing-sharedctypes-array/45694579?noredirect=1
        self.nn_info = sharedctypes.Array(ctypes.c_wchar,len(self.nn_paths)*k*self.n_patches*self.s_len)

        self.k = k
        print(f'Finish preparation of {self.__len__()} nns.')


    def load_data(self, paths):
        out = {}

        for p in tqdm(paths,desc='Preparing nn_paths. This might take a while...'):
            idx = int(p.split("-img")[-1].split(".")[0])
            try:
                with open(os.path.join(self.basepath,p), 'rb') as f:
                    nn_info = pickle.load(f)[self.n_patches_per_side]
            except EOFError as e:
                print('EOFError: ', e)
                print(f'Adding example with id {idx} to corrupts.')
            rimg_ids = nn_info['img_ids']
            p_coord = nn_info['patch_coords']


            for patch_id, nns in enumerate(rimg_ids):
                for nn_id,img_id in enumerate(nns):
                    # rimg_ids.add(img_id)
                    q_info = {'p_coords': [p_coord[patch_id, nn_id]],
                              'qid': [idx],
                              'nnid': [nn_id],
                              'patch_id': [patch_id]}
                    if img_id not in out:
                        out.update({img_id:q_info})
                    else:
                        for key in q_info:
                            out[img_id][key].extend(q_info[key])


        return out

    def save_metafile(self):
        to_save = {}
        for i in range(0,len(self.nn_info),self.s_len):
            qid = i // (self.k*self.n_patches*self.s_len)
            patch_id = (i % (self.k*self.n_patches*self.s_len)) // (self.k*self.s_len)
            nn_idXlen = (i - (qid*self.n_patches + patch_id)*self.k*self.s_len)
            assert nn_idXlen % self.s_len == 0
            nn_id = nn_idXlen // self.s_len

            fname = self.nn_info[i:i+self.s_len]

            if qid not in to_save:
                new_entry = np.full((self.k*self.n_patches,),'',dtype=object)
                new_entry[patch_id*self.k+nn_id] = fname
                to_save[qid] = new_entry
            else:
                to_save[qid][patch_id*self.k+nn_id] = fname

        print(to_save)
        savename = os.path.join(self.basepath,f'nns-{self.n_patches}_patches')
        # if part is not None:
        #     savename+=f'_part{part}'
        with open(savename+'.p','wb') as f:
            pickle.dump(to_save,f,protocol=pickle.HIGHEST_PROTOCOL)

        if len(self.corrupts) > 0:
            with open(os.path.join(self.basepath,'corrupts.txt'),'w') as f:
                f.write(self.corrupts)

    def __getitem__(self, idx):
        # info = self.labels[idx]

        q_info = self.q_info[idx]
        data = self.data[idx]
        rimg = data['image']
        if isinstance(rimg, torch.Tensor):
            rimg = rimg.numpy()
        for nnid, patch_id, p_coords, qid in zip(q_info['nnid'],q_info['patch_id'],q_info['p_coords'],q_info['qid'],):
            patch = rimg[p_coords[1]:p_coords[3],p_coords[0]:p_coords[2]]
            patch = (np.clip(rescale(patch),0,1)*255.).astype(np.uint8)

            relname = self.nn_paths[qid]
            fname = os.path.join(self.basepath, relname)

            self.save_single_patches(p_coords,idx,patch,fname,patch_id,nnid,qid)

        # create dummy output
        return {'image': torch.randn((128,128,3))}

    def __len__(self):
        return len(self.q_info)

    def save_single_patches(self,p_coords, rimg_id, patch, fp, patch_id, nnid, qid):
        basepath = '/'.join(fp.split('/')[:-2])
        patch_dir = os.path.join(basepath, 'nn_patches')
        os.makedirs(patch_dir, exist_ok=True)

        # these are overall 16 + 19 + 4 = 39 characters
        name = f'{rimg_id:09d}-patch_' + '-'.join([f'{c:04d}' for c in p_coords]) + '.png'
        # overall 50 characters
        relname = 'nn_patches'+ '/' + name
        assert len(relname) == self.s_len
        savename = os.path.join(patch_dir, name)
        if not os.path.isfile(savename):
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(savename, patch)

        try:
            self.nn_info[(qid*self.k*self.n_patches+patch_id*self.k+nnid)*self.s_len:
                        (qid*self.k*self.n_patches+patch_id*self.k+nnid+1)*self.s_len] = relname
        except ValueError as e:
            print(f'ValueERROR: ',e)
            print(f"Strange, sizes don't match :")
            print(f'Expected length: {(qid*self.k*self.n_patches+patch_id*self.k+nnid)*self.s_len - (qid*self.k*self.n_patches+patch_id*self.k+nnid+1)*self.s_len}')
            print(f'Actual name: {relname} of length {len(relname)}. Using last entry')
            i0 = qid * self.k * self.n_patches + patch_id * self.k + nnid
            self.nn_info[i0 * self.s_len : (i0 + 1) * self.s_len] = self.nn_info[(i0 - 1) * self.s_len : i0 * self.s_len]


class CustomSeqSampler(Sampler):
    def __init__(self, data_source: PatchSaverDataset):
        self.data_source = data_source

    def __iter__(self):
        valid_ids = list(self.data_source.q_info.keys())
        # random.shuffle(valid_ids)
        return iter(valid_ids)

    def __len__(self):
        return len(self.data_source)


def save_single_patches(p_coords, rimg_id,patch, fp, npps, patch_id, nnid,lock):
    basepath = '/'.join(fp.split('/')[:-2])
    patch_dir = os.path.join(basepath,'nn_patches')
    os.makedirs(patch_dir,exist_ok=True)

    name = f'{rimg_id:09d}-patch_' + '-'.join([str(c) for c in p_coords]) + '.png'
    relname = os.path.join('nn_patches',name)
    savename = os.path.join(patch_dir,name)
    if not os.path.isfile(savename):
        patch = cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
        cv2.imwrite(savename,patch)

    lock.acquire()
    with open(fp,'rb') as f:
        meta_data = pickle.load(f)


    md = {'patch_id': patch_id,
           'nn_id': nnid,
           'filepath': relname}



    if npps not in meta_data:
        meta_data[npps] = {'nn_patches': [md]}
    else:
        if 'nn_patches' in meta_data[npps]:
            meta_data[npps]['nn_patches'].append(md)
        else:
            meta_data[npps].update({'nn_patches':[md]})


    with open(fp,'wb') as f:
        pickle.dump(meta_data,f,protocol=pickle.HIGHEST_PROTOCOL)
    lock.release()


def load_data(paths, basepath, n_patches_per_side):
    out = {}

    for p in tqdm(paths,desc='Preparing nn_paths. This might take a while...'):
        idx = int(p.split("-img")[-1].split(".")[0])
        try:
            with open(os.path.join(basepath,p), 'rb') as f:
                nn_info = pickle.load(f)[n_patches_per_side]
        except EOFError as e:
            print('EOFError: ', e)
            print(f'Adding example with id {idx} to corrupts.')
        rimg_ids = nn_info['img_ids']
        p_coord = nn_info['patch_coords']


        for patch_id, nns in enumerate(rimg_ids):
            for nn_id,img_id in enumerate(nns):
                # rimg_ids.add(img_id)
                q_info = {'p_coords': [p_coord[patch_id, nn_id]],
                          'qid': [idx],
                          'nnid': [nn_id],
                          'patch_id': [patch_id]}
                if img_id not in out:
                    out.update({img_id:q_info})
                else:
                    for key in q_info:
                        out[img_id][key].extend(q_info[key])


    return out

def get_q_info(filepath, npps, n_loaders):
    with open(filepath, 'rb') as f:
        nn_paths = pickle.load(f)

    nnps = list(nn_paths.values())
    # n_loaders = min(n_loaders, len(nnps))
    load_fn = partial(load_data,basepath = '/'.join(filepath.split('/')[:-1]),n_patches_per_side = npps)
    nnps = parallel_data_prefetch(load_fn, nnps, n_loaders, target_data_type='dict')
    q_info = {}

    for subdict in tqdm(nnps, desc='Merging subdicts...'):
        for key in subdict:
            if key in q_info:
                for kkey in subdict[key]:
                    if kkey in q_info[key]:
                        q_info[key][kkey].extend(subdict[key][kkey])
                    else:
                        q_info[key][kkey] = subdict[key][kkey]
            else:
                q_info[key] = {kkey: subdict[key][kkey] for kkey in subdict[key]}

    return q_info

def save_pkl(filepath, save_it, npatches_perside, corrupts, i, j, start_id, dset_batch_size):
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                old_one = pickle.load(f)

            old_one.update({npatches_perside: save_it[npatches_perside]})
            with open(filepath, 'wb') as f:
                pickle.dump(old_one, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'ERROR: {e.__class__.__name__} : ', e)
            if npatches_perside == 1:
                print(f'Overwriting id {start_id + i * dset_batch_size + j} as it is corrupt.')
                with open(filepath, 'wb') as f:
                    pickle.dump(save_it, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                corrupts.add(start_id + i * dset_batch_size + j)
                print(f'Adding id {start_id + i * dset_batch_size + j} to corrupts.')
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(save_it, f, protocol=pickle.HIGHEST_PROTOCOL)

    return corrupts

def search_nns(dataset_builder, qloader, device='cuda', mode='img', save=False, npatches_perside=None, base_savedir=None, nn_paths=None, corrupts=None, start_id=0, max_its=None):
    assert dataset_builder.searcher is not None
    dset_batch_size = qloader.batch_size

    if save:
        assert base_savedir is not None
        assert npatches_perside is not None
        assert os.path.isdir(os.path.join(base_savedir,'embeddings'))
        if nn_paths is None:
            nn_paths = {}

        if corrupts is None:
            corrupts = []

    ns = 0

    return_ids = {}

    for i, batch in enumerate(tqdm(qloader, desc='Searching nns and saving embeddings', total=len(qloader) if max_its is None else max_its)):
        if max_its is not None and i >= max_its:
            break

        query = batch['patches'].to(device) if mode == 'img' else batch['caption']
        if isinstance(query, torch.Tensor):
            b, n, *_ = query.shape
            query = rearrange(query, 'b n h w c -> (b n) h w c')
        else:
            b, n = len(query), 1

        caption_sim = mode == 'text'
        results = dataset_builder.search_k_nearest(query, visualize=False, is_caption=caption_sim)

        if save:
            results = {
                key: results[key].reshape(b, n,*results[key].shape[1:])
                     if isinstance(results[key], np.ndarray) else
                     results[key]
                for key in results
            }

            for j in range(len(results['embeddings'])):
                filename = f'embeddings/{dataset_builder.k}_nns-img{start_id + i * dset_batch_size + j:09d}.p'
                filepath = os.path.join(base_savedir, filename)
                save_it = {npatches_perside: {
                    'embeddings': results['embeddings'][j],
                    'img_ids': results['img_ids'][j],
                    'patch_coords': results['patch_coords'][j],
                    'nn_ids': results['nns'][j]
                }}
                corrupts = save_pkl(filepath=filepath,
                                    save_it=save_it,
                                    npatches_perside=npatches_perside,
                                    corrupts=corrupts,
                                    i=i,j=j,
                                    start_id=start_id,
                                    dset_batch_size=dset_batch_size)

                nn_paths.update({start_id + i * dset_batch_size + j: filename})
                ns += 1
        else:
            ids, counts = np.unique(results['nns'], return_counts=True)
            for id_,n in zip(ids,counts):
                if id_ in return_ids:
                    return_ids[int(id_)] += n
                else:
                    return_ids[int(id_)] = n

    if save:
        return nn_paths
    else:
        return return_ids


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt = parser.parse_args()

    if opt.only_patches and opt.only_neighbors:
        raise ValueError('both --only_patches and --only_neighbors options are selected')


    config = OmegaConf.load(opt.rconfig)
    query_config = OmegaConf.load(opt.qconfig).data
    query_config = query_config.params[opt.split]

    # build retrieval dataset and train searcher
    dataset_builder: DatasetBuilder = instantiate_from_config(config.builder)

    # setting some paths
    base_savedir = os.path.join(
        opt.nns_savedir,
        f'{dataset_builder.data_pool["embedding"].shape[0]}p-{dataset_builder.retriever_name}_{dataset_builder.patch_size}@{dataset_builder.dset_name}',
        f'{query_config.params.dset_config.target.split(".")[-1]}',
    )

    print(f'Base savedir is {base_savedir}')

    assert opt.parts >= 1, 'Specified number of parts for the subset must be greater or equal 1'

    cfile_name = 'corrupts'
    mfile_name =  'nn_paths'
    if opt.parts> 1:
        mfile_name+=f'_p{opt.part_no}'
        cfile_name +=f'_p{opt.part_no}'

    mfile_name += '.p'
    cfile_name += '.txt'
    meta_filepath = os.path.join(base_savedir, mfile_name)
    c_filepath = os.path.join(base_savedir, cfile_name)

    if opt.mode == 'text':
        print('*'*100)
        print('Setting n_patches per side to 1 as other options not supported until now')
        print('*' * 100)
        opt.log_max_np = 0

    if not opt.only_patches:
        dataset_builder.build_data_pool()
        dataset_builder.load_embeddings()
        dataset_builder.train_searcher()

        nns = {}
        id_key = 0
        corrupts = set()

        device = next(dataset_builder.retriever.parameters()).device
        nn_paths = {}
        for n_p in range(opt.log_max_np+1):
            npatches_perside = 2**n_p
            n_patches = npatches_perside ** 2
            print(f'computing {dataset_builder.k} nns for {n_patches} patches per Image.')
            query_config.params['n_patches_per_side'] = npatches_perside
            query_dataset = instantiate_from_config(query_config)
            start_id = 0


            if opt.parts > 1:
                print(f'Dividing dataset into {opt.parts} subsets. Current run calculates NNs for subset #{opt.part_no}')
                n_examples = len(query_dataset)
                for i,sub_ids in enumerate(np.array_split(np.arange(n_examples),opt.parts)):
                    if i+1 == opt.part_no:
                        start_id = int(sub_ids[0])
                        print(f'Using id range [{start_id},{int(sub_ids[-1])})')
                        query_dataset=Subset(query_dataset,sub_ids)
                        break

            embeddings_savedir = os.path.join(base_savedir, 'embeddings')
            os.makedirs(base_savedir, exist_ok=True)
            os.makedirs(embeddings_savedir,exist_ok=True)


            dset_batch_size = opt.batch_size // n_patches
            qloader = DataLoader(
                query_dataset,
                batch_size=dset_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=opt.n_workers
            )

            nn_paths= search_nns(
                dataset_builder=dataset_builder,
                qloader=qloader,
                mode=opt.mode,
                save=True,
                npatches_perside=npatches_perside,
                base_savedir=base_savedir,
                nn_paths=nn_paths,
                corrupts=corrupts,
                start_id=start_id,
                device=device,
            )

            with open(meta_filepath, 'wb') as f:
                pickle.dump(nn_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(c_filepath,'w') as fc:
            for item in corrupts:
                fc.write(str(item)+'\n')


    if (opt.parts == 1 or opt.only_patches) and not opt.only_neighbors:
        print(f'loading precomputed nns from "{meta_filepath}"')
        assert os.path.isfile(meta_filepath)
        with open(meta_filepath,'rb') as f:
            nn_paths = pickle.load(f)

        for n_p in range(opt.log_max_np+1):
            print('loading nns as image patches and saving them with the remaining data')
            npatches_perside = 2 ** n_p
            n_patches = npatches_perside**2

            q_info = get_q_info(meta_filepath, npatches_perside, n_loaders=min(cpu_count(), opt.n_workers))
            k_patch = query_config.params.k
            pset = PatchSaverDataset(
                dataset_builder.patch_dset,
                meta_filepath,
                npatches_perside,
                k=k_patch,
                q_info=q_info
            ) # min(100,cpu_count()
            qloader = DataLoader(
                pset,
                batch_size=opt.batch_size,
                sampler=CustomSeqSampler(pset),
                drop_last=False,
                num_workers=opt.n_workers
            )

            for i, batch in enumerate(tqdm(qloader, desc='Also saving nns as patches', total=len(qloader))):
                data = batch['image']

            print(f'saving nn patch paths for {npatches_perside} and {dataset_builder.k} nns.')
            pset.save_metafile()
            print('Finish saving')

        print('done')
