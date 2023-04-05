import os
from abc import abstractmethod
from functools import partial

import albumentations
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from ldm.modules.image_degradation import (degradation_fn_bsr,
                                           degradation_fn_bsr_light)

try:
    from rdm.data.openimages_bbox_helpers.openimages_bbox_base import \
        OpenImagesBBoxBase
except ImportError as e:
    print(e)
    print("### need to install torch-sparse.")
    class OpenImagesBBoxBase:
        pass

FULL_TRAIN_FILES = [f"trainfiles{i}.txt" for i in range(10)]
VALIDATION_FILES = "validationfiles.txt"
TEST_FILES = "testfiles.txt"


class FullOpenImagesBase(Dataset):
    def __init__(self, size=None, crop_size=None, interpolation="bicubic",
                 data_root="data/fullopenimages/", random_crop=True,
                 corrupt_files="data/unidentifiable_openimage_files.txt"):
        self.split = self.get_split()
        self.size = size
        self.crop_size = crop_size if crop_size is not None else size
        if self.size is not None: assert self.crop_size<=self.size
        self.data_files = {"train": FULL_TRAIN_FILES,  "validation": VALIDATION_FILES, "test": TEST_FILES}
        self.data_root = data_root
        print("Building Full OpenImages Dataset. Get a Coffee.")
        self.image_paths = list()
        if self.split == "train":
            for i in trange(len(FULL_TRAIN_FILES), desc="Constructing Train Data"):
                with open(os.path.join(self.data_root, self.data_files[self.split][i]), "r") as f:
                    for line in f:
                        self.image_paths.append(os.path.join(f"train{i}", line.rstrip()))

            # quicky and dirty clean-up:
            with open(corrupt_files, "r") as cf:
                corrupts = cf.read().splitlines()
                for corrupt in tqdm(corrupts, desc="Remove Corrupts"):
                    corrupt = "/".join(corrupt.split("/")[2:])
                    self.image_paths.remove(corrupt)

        else:
            with open(os.path.join(self.data_root, self.data_files[self.split]), "r") as f:
                for line in tqdm(f, desc="Constructing Validation Data"):
                    self.image_paths.append(os.path.join(self.split, line.rstrip()))

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": self.image_paths,
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        #if self.size is not None:
        self.interpolation = interpolation
        self.interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size if self.size is not None else self.crop_size,
                                                             interpolation=self.interpolation)
        self.center_crop = not random_crop
        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)

        self.unident_files = "unidentifiable_openimage_files.txt"  # only debug

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None or (self.crop_size is not None and min(image.shape[0], image.shape[1]) < self.crop_size):
            image = self.image_rescaler(image=image)["image"]
        if self.cropper is not None and self.crop_size is not None:
            image = self.cropper(image=image)["image"]
        processed = {"image": image}
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        return example

class SemanticOpenImages(Dataset):

    def __init__(self, csv_root='data/semantic_openimages', image_root='data/openimages/',size=None, crop_size=None,
                random_crop=True, interpolation="bicubic",segmentation_to_float=True, crop_around_roi=True):
        super().__init__()
        print(f'Building {self.__class__.__name__} dataset.')
        self.split=self.get_split()
        self.csv_root = csv_root
        self.image_root = os.path.join(image_root,self.split)
        self.segmentation_root = os.path.join(self.csv_root,self.split)
        print(f'Reading annotations file...This may take a while...')
        self.data_csv = pd.read_csv(os.path.join(self.csv_root,f'{self.split}-annotations-object-segmentation.csv')).to_dict()
        self.size = size
        self.seg_to_float = segmentation_to_float
        classid2text = pd.read_csv(os.path.join(self.csv_root,'class-descriptions-boxable.csv'))
        self.classid2text = {idx: t for idx,t in zip(classid2text['classid'],classid2text['description'])}
        self.crop_around_roi = crop_around_roi


        with open(os.path.join(self.csv_root,'segmentation_classes.txt'),'r') as f:
            # add one as id 0 is background
            self.classid2label = {line.rstrip():i+1 for i,line in enumerate(f.readlines())}

        # also add 1 for background class
        self.num_classes = len(self.classid2label) + 1
        self.class_labels = [self.classid2label[key] for key in self.data_csv['LabelName'].values()]

        self.labels = {
            'class_name': [self.classid2text[key] for key in self.data_csv['LabelName'].values()],
            'class_label': self.class_labels,
            'relative_file_path_': [os.path.join(fn+'.jpg') for fn in self.data_csv['ImageID'].values()],
            'file_path_': [os.path.join(self.image_root,fn+'.jpg') for fn in self.data_csv['ImageID'].values()],
            'segmentation_path_': [os.path.join(self.segmentation_root,fn) for fn in self.data_csv['MaskPath'].values()]
        }

        self._length = len(self.labels['file_path_'])

        size = None if size is not None and size <= 0 else size
        self.size = size
        self.crop_size = self.size if crop_size is None else crop_size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop



            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.preprocessor = self.cropper



    def __len__(self):
        return self._length

    def crop_img_and_mask_around_roi(self,image,mask):
        pass

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        mask = np.array(Image.open(example["segmentation_path_"]))
        segmentation = np.zeros(mask.shape, dtype=np.int)
        segmentation = np.where(mask,np.full_like(segmentation,example['class_label']),segmentation)

        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
            if self.crop_around_roi:
                processed = self.crop_img_and_mask_around_roi(image=image,
                                                              mask=segmentation)
            else:
                processed = self.preprocessor(image=image,
                                          mask=segmentation
                                          )
        else:
            processed = {"image": image,
                         "mask": segmentation
                         }
        segmentation = processed["mask"]
        onehot = np.eye(self.num_classes)[segmentation]
        example["segmentation"] = onehot
        example['mask'] = segmentation

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)

        if self.seg_to_float:
            example['segmentation'] = example['segmentation'].astype(np.float32)

        return example


    @abstractmethod
    def get_split(self):
        raise NotImplementedError()

class SemanticOpenImagesTrain(SemanticOpenImages):

    def __init__(self, size=None, crop_size=None, random_crop=True, interpolation="bicubic",segmentation_to_float32=True):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation,
                         segmentation_to_float = segmentation_to_float32
                         )

    def get_split(self):
        return 'train'


class SemanticOpenImagesValidation(SemanticOpenImages):

    def __init__(self, size=None, crop_size=None, random_crop=True, interpolation="bicubic",segmentation_to_float32=True):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation,
                         segmentation_to_float=segmentation_to_float32
                         )

    def get_split(self):
        return 'validation'

class SemanticOpenImagesTest(SemanticOpenImages):

    def __init__(self, size=None, crop_size=None, random_crop=True, interpolation="bicubic",segmentation_to_float32=True):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation,
                         segmentation_to_float=segmentation_to_float32
                         )

    def get_split(self):
        return 'test'


class SuperresOpenImages(Dataset):
    def __init__(self, size=None, crop_size=None, interpolation="bicubic",
                 degradation="bsrgan", downscale_f=4,
                 data_root="data/fullopenimages/", random_crop=True,
                 corrupt_files="data/unidentifiable_openimage_files.txt"):
        """
        :param size: setting size will resize the image before applying a crop
        :param crop_size:
        :param interpolation:
        :param degradation:
        :param downscale_f:
        :param data_root:
        :param random_crop:
        :param corrupt_files:
        """
        self.split = self.get_split()
        assert(size or crop_size)
        self.size = size
        self.crop_size = crop_size if crop_size is not None else size
        if self.size is not None: assert self.crop_size<=self.size
        assert((crop_size / downscale_f).is_integer())
        self.data_files = {"train": FULL_TRAIN_FILES,  "validation": VALIDATION_FILES, "test": TEST_FILES}
        self.data_root = data_root
        print("Building Full OpenImages Dataset. Get a Coffee.")
        self.image_paths = list()
        if self.split == "train":
            for i in trange(len(FULL_TRAIN_FILES), desc="Constructing Train Data"):
                with open(os.path.join(self.data_root, self.data_files[self.split][i]), "r") as f:
                    for line in f:
                        self.image_paths.append(os.path.join(f"train{i}", line.rstrip()))

            # quicky and dirty clean-up:
            with open(corrupt_files, "r") as cf:
                corrupts = cf.read().splitlines()
                for corrupt in tqdm(corrupts, desc="Remove Corrupts"):
                    corrupt = "/".join(corrupt.split("/")[2:])
                    self.image_paths.remove(corrupt)

        else:
            with open(os.path.join(self.data_root, self.data_files[self.split]), "r") as f:
                for line in tqdm(f, desc="Constructing Validation Data"):
                    self.image_paths.append(os.path.join(self.split, line.rstrip()))

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": self.image_paths,
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        if size:
            rescale_size = size
        else:
            rescale_size = self.crop_size

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=rescale_size, interpolation=cv2.INTER_CUBIC)

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            degradation_fn = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[degradation]

            self.degradation_process = albumentations.SmallestMaxSize(max_size=self.crop_size // downscale_f,
                                                                 interpolation=degradation_fn)

        self.center_crop = not random_crop

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)

        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)

        self.unident_files = "unidentifiable_openimage_files.txt"  # only debug

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        if self.size is not None or image.shape[0] < self.crop_size or image.shape[1] < self.crop_size:
            image = self.image_rescaler(image=image)["image"]

        if self.cropper is not None:
            image = self.cropper(image=image)["image"]

        LR_image = self.degradation_process(image=image)["image"]

        processed = {"image": image, "LR_image": LR_image}

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (processed["LR_image"]/127.5 - 1.0).astype(np.float32)

        return example


class SuperresOpenImagesTrain(SuperresOpenImages):
    def __init__(self, size=None, crop_size=None, random_crop=True,
                 interpolation="bicubic", degradation="bsrgan", downscale_f=4):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation,
                         degradation=degradation,
                         downscale_f=downscale_f
                         )

    def get_split(self):
        return 'train'


class SuperresOpenImagesValidation(SuperresOpenImages):
    def __init__(self, size=None, crop_size=None, random_crop=True,
                 interpolation="bicubic", degradation="bsrgan", downscale_f=4):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation,
                         degradation=degradation,
                         downscale_f=downscale_f
                         )

    def get_split(self):
        return 'validation'


class SuperresOpenImagesxFaces(SuperresOpenImages):
    def __init__(self, size=None, crop_size=None, interpolation="bicubic",
                 degradation="bsrgan", downscale_f=4,
                 data_root="data/fullopenimages/", random_crop=True,
                 corrupt_files="data/unidentifiable_openimage_files.txt",
                 colorize=False):
        super().__init__(size=size,
                         crop_size=crop_size,
                         interpolation=interpolation,
                         degradation=degradation,
                         downscale_f=downscale_f,
                         data_root=data_root,
                         random_crop=random_crop,
                         corrupt_files = corrupt_files,
                         )
        self.face_size = 256  # todo
        self.alpha_blurred_circle = make_blur_circle(size=(self.face_size, self.face_size), filter_sizer=16) # todo
        self.p_face = 0.2 #todo
        self.colorize = colorize

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        if self.size is not None or image.shape[0] < self.crop_size or image.shape[1] < self.crop_size:
            image = self.image_rescaler(image=image)["image"]

        if self.cropper is not None:
            image = self.cropper(image=image)["image"]

        # This is the part where we put a flying face
        h, w = self.crop_size, self.crop_size # TODO
        min_size, max_size = w//8, w  # TODO

        image = Image.fromarray(image)
        if np.random.uniform(0, 1) < self.p_face:
            idx = np.random.randint(len(self.face_data))
            face = self.face_data[idx]["image"]
            face = ((face + 1) * 127.5).astype(np.uint8)
            face = Image.fromarray(face)
            face.putalpha(self.alpha_blurred_circle)
            pos = (int(np.random.uniform(0, w)), int(np.random.uniform(0, h)))
            size = int(np.random.uniform(min_size, max_size))
            put_face(face, pos, size)

        image = np.array(image)
        LR_image = self.degradation_process(image=image)["image"]

        processed = {"image": image, "LR_image": LR_image}

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (processed["LR_image"]/127.5 - 1.0).astype(np.float32)

        if self.colorize:
            # for colorization training
            # transformation according to https://github.com/ericsujw/InstColorization/blob/master/data/aligned_dataset.py
            gray_image = TF.to_tensor(image)
            gray_image = TF.normalize(gray_image, mean=[.5, .5, .5], std=[.5, .5, .5])
            gray_image = gray_image[0, ...] * 0.299 + gray_image[1, ...] * 0.587 + gray_image[2, ...] * 0.114
            gray_image = gray_image[..., None].numpy()
            example['gray_image'] = gray_image

        return example


class FullOpenImagesTrain(FullOpenImagesBase):
    def __init__(self, size=None, crop_size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation
                         )

    def get_split(self):
        return "train"


class FullOpenImagesValidation(FullOpenImagesBase):
    def __init__(self, size=None, crop_size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(size=size,
                         crop_size=crop_size,
                         random_crop=random_crop,
                         interpolation=interpolation
                         )

    def get_split(self):
        return "validation"



class OpenImagesBBoxTrain(OpenImagesBBoxBase):

    def __init__(self, size, encode_bbox=True, datapath='data/openimages',n_max_samples=-1,
                 random_flip=True,crop_type='random-2d', keys=None,**kwargs):
        if keys is None:
            keys = ['image', 'coordinates_bbox', 'bbox_img', 'image_with_bboxes']
        self.size = size
        super().__init__(data_path=datapath,split='train',keys=keys,
                         target_image_size=size,no_max_samples=n_max_samples,
                         crop_method=crop_type,random_flip=random_flip,
                         encode_crop=encode_bbox,**kwargs)

class OpenImagesBBoxValidation(OpenImagesBBoxBase):

    def __init__(self, size, encode_bbox=True, datapath='data/openimages', n_max_samples=-1,
                 random_flip=False, crop_type='center', keys=None, **kwargs):
        if keys is None:
            keys = ['image', 'coordinates_bbox', 'bbox_img', 'image_with_bboxes']
        self.size = size
        super().__init__(data_path=datapath, split='validation', keys=keys,
                         target_image_size=size, no_max_samples=n_max_samples,
                         crop_method=crop_type, random_flip=random_flip,
                         encode_crop=encode_bbox, **kwargs)


TXT_TRAIN = "data/openimages_annotated_train.txt"
TXT_VAL = "data/openimages_annotated_validation.txt"

class OpenImagesBase(Dataset):
    # Note: These are the images from the annotated subset of OpenImages (~ 1.8 M images)
    def __init__(self, size=None, random_crop=False, interpolation="bicubic",
                 data_root="data/openimages/", only_crop_size=-1):
        self.split = self.get_split()
        self.data_csv = {"train": TXT_TRAIN,  "validation": TXT_VAL}[self.split]
        self.data_root = os.path.join(data_root, self.split)
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        size = None if size is not None and size<=0 else size
        self.size = size if only_crop_size == -1 else None
        self.only_crop_size = only_crop_size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)

            self.preprocessor = self.cropper
        if only_crop_size > -1:
            self.preprocessor = albumentations.RandomCrop(height=only_crop_size, width=only_crop_size)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image)
        elif self.only_crop_size > -1:
            processed = self.preprocessor(image=image)
        else:
            processed = {"image": image,
                         }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        return example


class OpenImagesTrain(OpenImagesBase):
    def __init__(self, size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(size=size,
                         random_crop=random_crop,
                         interpolation=interpolation
                         )

    def get_split(self):
        return "train"


class OpenImagesValidation(OpenImagesBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(size=size,
                         random_crop=random_crop,
                         interpolation=interpolation
                         )

    def get_split(self):
        return "validation"


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from tqdm import trange

    # d0 = FullOpenImagesValidation(crop_size=256)
    # for i in trange(1009):
    #     ex = d0[i]
    #     img = ex["image"]
    # d1 = SemanticOpenImagesTrain(size=256)
    # print("construced train set.")
    # print(f"length of train split: {len(d1)}")
    d2= OpenImagesBBoxValidation(size=256)
    print("construced train set.")
    print(f"length of train split: {len(d2)}")
    # d3 = OpenImagesBBoxValidation(size=256,encode_bbox=False)
    # print('constructed validation set.')
    # print(f"length of validation split: {len(d3)}")
    # ex1 = d1[0]
    # if issubclass(d2.__class__, OpenImagesBBoxBase):
    #
    #     print(f'Category numbers: {d2.unique_category_numbers}')
    #
    #     testpath = 'test_oimages'
    #     os.makedirs(testpath,exist_ok=True)
    #
    #     num_grids = 20
    #     loader = DataLoader(d2,batch_size=12,shuffle=True)
    #
    #     for gridn, batch in enumerate(loader):
    #
    #         if gridn >= num_grids:
    #             break
    #         debug_imgs = batch['image_with_bboxes'].permute(0,3,1,2)
    #
    #         save_image(debug_imgs,fp=os.path.join(testpath,f'test_grid_{gridn}.png'),nrow=4,normalize=True,padding=4)
    #
    # else:
    ex2 = d2[0]
    # ex3 = d3[0]
    # print(ex1["image"].shape)
    print(ex2["image"].shape)
    # print(ex3["image"].shape)
    # print(ex1["segmentation"].shape)
    print(ex2["bbox_img"].shape)
    print(ex2['coordinates_bbox'].shape)
    print(ex2['coordinates_bbox'])
    # print(ex3["coordinates_bbox"])

    image = ((ex2['image'].numpy()+1.)*127.5).astype(np.uint8)
    bbox_img = ((ex2['bbox_img'].numpy()+1.)*127.5).astype(np.uint8)
    print(ex2['annotations'])
    # class_name = ex2['class_name']
    # segmentation = ex2['mask'].astype(np.bool)

    # color_mask = np.zeros_like(image)
    # color_mask[:,:,0] = 255
    # seg_vis = np.where(segmentation[...,None],color_mask,np.zeros_like(image))

    overlay = cv2.addWeighted(image,0.8,bbox_img,0.2,0.)
    overlay = cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'overlay.png',overlay)
    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'bbox_img.png',bbox_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'image.png', image)
    print("done.")

    # d1 = FullOpenImagesTrain(size=256)
    # print("construced train set.")
    # print(f"length of train split: {len(d1)}")
    # d2 = FullOpenImagesValidation(size=256)
    # ex1 = d1[0]
    # ex2 = d2[0]
    # print(f"length of val split: {len(d2)}")
    # print(ex1["image"].shape)
    # print(ex2["image"].shape)
    # print("done.")
