import math
import random
import warnings
from collections import defaultdict
from contextlib import suppress
from csv import DictReader
from csv import reader as TupleReader
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image as pil_img
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from rdm.data.openimages_bbox_helpers.category_mappings import \
    open_images_unify_categories_for_coco
from rdm.data.openimages_bbox_helpers.openimages_builder import (
    CoordinatesBoundingBoxConditionalBuilder, RescaledAnnotationsBuilder)
from rdm.data.openimages_bbox_helpers.openimages_builderutils import (
    Annotation, BoundingBox, Category, CenterCropReturnCoordinates,
    CropMethodType, IntelligentRandomCropReturnCoordinates,
    Random2dCropReturnCoordinates, RandomCropReturnCoordinates,
    RandomHorizontalFlipReturnValue, convert_pil_to_tensor, draw_bboxes,
    draw_colorized_bboxes)
from rdm.data.openimages_bbox_helpers.selected_categories import \
    top_300_classes_plus_coco_compatibility

OPEN_IMAGES_STRUCTURE = {
    'train': {
        'top_level': '',
        'class_descriptions': 'class-descriptions-boxable.csv',
        'annotations': 'oidv6-train-annotations-bbox.csv',
        'file_list': 'train-images-boxable.csv',
        'files': 'train'
    },
    'validation': {
        'top_level': '',
        'class_descriptions': 'class-descriptions-boxable.csv',
        'annotations': 'validation-annotations-bbox.csv',
        'file_list': 'validation-images.csv',
        'files': 'validation'
    },
    'test': {
        'top_level': '',
        'class_descriptions': 'class-descriptions-boxable.csv',
        'annotations': 'test-annotations-bbox.csv',
        'file_list': 'test-images.csv',
        'files': 'test'
    }
}

def load_annotations(descriptor_path: Path, min_object_area: float, category_mapping: Dict[str, str],
                     category_no_for_id: Dict[str, int]) -> Dict[str, List[Annotation]]:
    annotations: Dict[str, List[Annotation]] = defaultdict(list)
    with open(descriptor_path) as file:
        # reader = zip(DictReader(file), range(20000))
        # for i, (row, _) in tqdm(enumerate(reader), total=20000, desc='Loading OpenImages annotations'):
        reader = DictReader(file)
        for i, row in tqdm(enumerate(reader), total=14620000, desc='Loading OpenImages annotations --> Get yourself a coffee (or two) in the meantime...'):
            width = float(row['XMax']) - float(row['XMin'])
            height = float(row['YMax']) - float(row['YMin'])
            area = width * height
            category_id = row['LabelName']
            if category_id in category_mapping:
                category_id = category_mapping[category_id]
            if area >= min_object_area and category_id in category_no_for_id:
                annotations[row['ImageID']].append(
                    Annotation(
                        id=i,
                        image_id=row['ImageID'],
                        source=row['Source'],
                        category_id=category_id,
                        category_no=category_no_for_id[category_id],
                        confidence=float(row['Confidence']),
                        bbox=(float(row['XMin']), float(row['YMin']), width, height),
                        area=area,
                        is_occluded=bool(int(row['IsOccluded'])),
                        is_truncated=bool(int(row['IsTruncated'])),
                        is_group_of=bool(int(row['IsGroupOf'])),
                        is_depiction=bool(int(row['IsDepiction'])),
                        is_inside=bool(int(row['IsInside']))
                    )
                )
    return dict(annotations)


def load_image_ids(csv_path: Path) -> List[str]:
    with open(csv_path) as file:
        reader = DictReader(file)
        return [row['image_name'] for row in reader]


def load_categories(csv_path: Path) -> Dict[str, Category]:
    with open(csv_path) as file:
        reader = TupleReader(file)
        return {row[0]: Category(id=row[0], name=row[1], super_category=None) for row in reader}


class OpenImagesBBoxBase(Dataset):

    def __init__(self,data_path: Union[str, Path], split: str, keys: List[str], target_image_size: int,
                 no_max_samples: int,
                 crop_method:CropMethodType,
                 random_flip: bool,
                 encode_crop:bool,
                 no_tokens=8192,
                 category_allow_list_target=top_300_classes_plus_coco_compatibility,
                 category_mapping_target=open_images_unify_categories_for_coco,
                 min_object_area=0.0001,
                 min_objects_per_image=2,
                 max_objects_per_image=30,
                 relation_type='SixRelation',
                 crop_coordinates_min_area=0.0001,
                 random_object_order=True,
                 use_group_parameter=True,
                 use_additional_parameters=True,
                 tok_no_max_relations=30,
                 tok_use_separator=False,
                 fill_bbox=False,
                 seed=None,
                 load_stuff=False
                 ):
        super().__init__()
        self.config = None
        self.split = split
        self.paths = self._build_paths(data_path)
        self.keys = keys
        self.load_annotation_type = {'things': True, 'stuff':load_stuff}
        self.no_max_samples = no_max_samples
        self.annotations = None
        self.image_ids = None
        self.image_descriptions = {}
        self.categories = None
        self.category_ids = None
        self.category_number = None
        self.category_allow_list = None
        self.fill_bbox = fill_bbox
        self.seed = seed
        if seed is not None:
            print(f'{self.__class__.__name__} setting random seed to {self.seed}')

        self.size = target_image_size


        if category_allow_list_target:
            allow_list = category_allow_list_target
            self.category_allow_list = {name for name, _ in allow_list}
        self.category_mapping = {}
        if category_mapping_target:
            if isinstance(category_mapping_target,str) and category_mapping_target.endswith('.yaml'):
                self.category_mapping = OmegaConf.load(category_mapping_target)
            else:
                self.category_mapping = category_mapping_target
        self.conditional_builders = {
            'annotations': RescaledAnnotationsBuilder(self.no_classes, relation_type, crop_coordinates_min_area,
                                                      random_object_order, no_tokens, use_group_parameter,
                                                      use_additional_parameters, max_objects_per_image),
            'coordinates_bbox':
                CoordinatesBoundingBoxConditionalBuilder(self.no_classes, relation_type, crop_coordinates_min_area,
                                                         tok_no_max_relations, tok_use_separator, random_object_order,
                                                         no_tokens, use_group_parameter, use_additional_parameters,
                                                         encode_crop)
        }
        self.transform_functions = self.setup_transform(target_image_size, crop_method, random_flip,
                                                        min_objects_per_image, max_objects_per_image,
                                                        crop_coordinates_min_area)


        self._load(min_object_area,min_objects_per_image,max_objects_per_image,crop_method)

        assert self.n_categories == self.no_classes

    def _load(self, min_object_area: float, min_objects_per_image: int, max_objects_per_image: int, crop_method: CropMethodType):
        self.categories = load_categories(self.paths['class_descriptions'])
        self.filter_categories()
        self.setup_category_id_and_number()
        self.unique_category_numbers = np.unique(list(self.category_number.values()))
        self.n_categories = self.unique_category_numbers.shape[0]

        annotations_unfiltered = load_annotations(self.paths['annotations'], min_object_area, self.category_mapping,
                                                  self.category_number)
        if crop_method == 'intelligent':
            max_objects_per_image = math.inf
        self.annotations = self.filter_annotations(annotations_unfiltered, min_object_area, min_objects_per_image,
                                                   max_objects_per_image)

        self.limit_no_samples()
        self.image_descriptions = {}
        self.clean_up_annotations_and_image_descriptions()

    @staticmethod
    def setup_transform(target_image_size: int, crop_method: CropMethodType, random_flip: bool,
                        min_objects_per_image: int, max_objects_per_image: int, crop_coordinates_min_area: float):
        transform_functions = []
        if target_image_size is not None:
            if crop_method == 'none':
                transform_functions.append(transforms.Resize((target_image_size, target_image_size)))
            elif crop_method == 'center':
                transform_functions.extend([
                    transforms.Resize(target_image_size),
                    CenterCropReturnCoordinates(target_image_size)
                ])
            elif crop_method == 'random':
                transform_functions.extend([
                    transforms.Resize(target_image_size),
                    RandomCropReturnCoordinates(target_image_size)
                ])
            elif crop_method == 'intelligent':  # working title
                transform_functions.extend([
                    IntelligentRandomCropReturnCoordinates(target_image_size, min_objects_per_image, max_objects_per_image,
                                                           crop_coordinates_min_area, False),
                    transforms.Resize((target_image_size, target_image_size))
                ])
            elif crop_method == 'random-2d':
                transform_functions.extend([
                    Random2dCropReturnCoordinates(target_image_size),
                    transforms.Resize(target_image_size)
                ])
            else:
                raise ValueError(f'Receive invalid crop method [{crop_method}].')
            if random_flip:
                transform_functions.append(RandomHorizontalFlipReturnValue())
        # noinspection PyTypeChecker
        transform_functions.append(transforms.Lambda(lambda x: x / 127.5 - 1.))
        return transform_functions

    @staticmethod
    def _load_image_from_disk(path: Path):
        return pil_img.open(path).convert('RGB')

    def _build_paths(self, top_level: Union[str, Path]) -> Dict[str, Path]:
        top_level = Path(top_level)
        sub_paths = {name: top_level.joinpath(sub_path) for name, sub_path in self.get_path_structure().items()}
        for path in sub_paths.values():
            if not path.exists():
                raise FileNotFoundError(f'{type(self).__name__} data structure error: [{path}] does not exist.')
        return sub_paths

    @staticmethod
    def filter_annotations(all_annotations: Dict[str, List[Annotation]], min_object_area: float,
                           min_objects_per_image: int, max_objects_per_image: int) -> Dict[str, List[Annotation]]:
        filtered = {}
        for image_id, annotations in all_annotations.items():
            annotations_with_min_area = [a for a in annotations if a.area > min_object_area]
            if min_objects_per_image <= len(annotations_with_min_area) <= max_objects_per_image:
                filtered[image_id] = annotations_with_min_area
        return filtered

    def filter_categories(self) -> None:
        if self.category_allow_list:
            self.categories = {id_: cat for id_, cat in self.categories.items() if cat.name in self.category_allow_list}
        if self.category_mapping:
            self.categories = {id_: cat for id_, cat in self.categories.items() if cat.id not in self.category_mapping}

    def setup_category_id_and_number(self) -> None:
        self.category_ids = list(self.categories.keys())
        self.category_ids.sort()
        if '/m/01s55n' in self.category_ids:
            self.category_ids.remove('/m/01s55n')
            self.category_ids.append('/m/01s55n')
        self.category_number = {category_id: i for i, category_id in enumerate(self.category_ids)}
        if self.category_allow_list is not None and self.category_mapping is None \
                and len(self.category_ids) != len(self.category_allow_list):
            warnings.warn('Unexpected number of categories: Mismatch with category_allow_list. '
                          'Make sure all names in category_allow_list exist.')

    def limit_no_samples(self) -> None:
        self.image_ids = list(self.annotations.keys())
        if self.no_max_samples > -1:
            if self.seed is not None:
                rd = random.Random(self.seed)
            else:
                rd = random.Random()
            self.image_ids = rd.sample(self.image_ids, self.no_max_samples)

        print(f'{self.__class__.__name__} consists of {len(self.image_ids)} examples.')
        self.image_ids.sort()

    def clean_up_annotations_and_image_descriptions(self) -> None:
        image_id_set = set(self.image_ids)
        self.annotations = {k: v for k, v in self.annotations.items() if k in image_id_set}
        self.image_descriptions = {k: v for k, v in self.image_descriptions.items() if k in image_id_set}

    def image_transform(self, x: Tensor, annotations: List[Annotation]) -> (Optional[BoundingBox], bool, Tensor):
        crop_bbox = None
        flipped = False
        for t in self.transform_functions:
            if isinstance(t, (RandomCropReturnCoordinates, CenterCropReturnCoordinates, Random2dCropReturnCoordinates)):
                crop_bbox, x = t(x)
            elif isinstance(t, IntelligentRandomCropReturnCoordinates):
                crop_bbox, x = t(x, annotations)
            elif isinstance(t, RandomHorizontalFlipReturnValue):
                flipped, x = t(x)
            else:
                x = t(x)
        return crop_bbox, flipped, x

    def __getitem__(self, n: int) -> Dict[str, Any]:
        image_id = self.get_image_id(n)
        sample = self.get_image_description(image_id)
        annotations = self.get_annotations(image_id)
        sample["annotations"] = annotations
        cat_ids = [a.category_id for a in annotations]
        cat_nrs = [self.get_category_number(cid) for cid in cat_ids]
        human_labels = [self.get_textual_label(cid) for cid in cat_ids]

        crop_bbox = None
        flipped = None

        if 'image' in self.keys or 'bbox_img' in self.keys:
            sample['image_path'] = str(self.get_image_path(image_id))
            sample['image'] = self._load_image_from_disk(sample['image_path'])
            if 'bbox_img' in self.keys:
                bboxes = [a.bbox for a in annotations]
                categories = [a.category_no for a in annotations]
                # for debug
                sample['image_with_bboxes'] = draw_bboxes(sample['image'], bboxes,cat_nrs,self.n_categories,human_labels)

                sample['bbox_img'] = draw_colorized_bboxes(sample['image'].size,bboxes,categories,self.no_classes,self.fill_bbox)


                for key in ['image', 'image_with_bboxes' , 'bbox_img',]:
                    sample[key] = convert_pil_to_tensor(sample[key]).unsqueeze(0)
                images = torch.cat((sample['image'], sample['image_with_bboxes'],sample['bbox_img']), dim=0)
                crop_bbox, flipped, images = self.image_transform(images, annotations)
                sample['image'] = images[0].squeeze().permute(1,2,0)
                sample['image_with_bboxes'] = images[1].squeeze().permute(1,2,0)
                sample['bbox_img'] = images[2].squeeze().permute(1,2,0)
            else:
                sample['image'] = convert_pil_to_tensor(sample['image']).permute(1,2,   0)
                crop_bbox, flipped, sample['image'] = self.image_transform(sample['image'], annotations)

        for conditional, builder in self.conditional_builders.items():
            if conditional in self.keys:
                sample[conditional] = builder.build(annotations, crop_bbox, flipped)



        # if self.target_logits_path:
        #     logits_file_name = self.target_logits_path.joinpath(self.split).joinpath(sample['file_name'])
        #     sample['logits'] = load_logits_from_topk(self.no_tokens, str(logits_file_name))

        if self.keys:
            # only return specified keys
            sample = {key: sample[key] for key in self.keys}
        return sample

    def __len__(self):
        return len(self.image_ids)

    def init_load_annotation_type(self, load_annotation_type: Dict[str, bool]) -> Dict[str, bool]:
        if 'things' not in load_annotation_type:
            load_annotation_type['things'] = True
        if 'stuff' in load_annotation_type and load_annotation_type['stuff'] is True:
            warnings.warn('Tried to load `stuff` categories but `Open Images` dataset does not have any. Deactivating.')
        load_annotation_type['stuff'] = False

        if not (load_annotation_type['things'] or load_annotation_type['stuff']):
            raise ValueError('Specified no categories. Specify to load `things` categories!')
        return load_annotation_type

    @property
    def base_no_classes(self) -> int:
        return 600

    @property
    def no_classes(self) -> int:
        with suppress(TypeError):
            return len(self.categories)
        with suppress(TypeError):
            return len(self.category_allow_list)
        return self.base_no_classes

    def get_path_structure(self) -> Dict[str, str]:
        if self.split not in OPEN_IMAGES_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for Open Images data.]')
        return OPEN_IMAGES_STRUCTURE[self.split]

    def get_image_id(self, n: int) -> str:
        return self.image_ids[n]

    def get_image_path(self, image_id: str) -> Path:
        return self.paths['files'].joinpath(f'{image_id:0>16}.jpg')

    def get_image_description(self, image_id: str) -> Dict[str, Any]:
        return {'file_path': str(self.get_image_path(image_id))}

    def get_annotations(self, image_id: str) -> List[Annotation]:
        return self.annotations[image_id]

    def get_textual_label(self, category_id: str) -> str:
        return self.categories[category_id].name

    def get_category_number(self, category_id: str) -> int:
        return self.category_number[category_id]

    def get_category_id(self, category_no: int) -> str:
        return self.category_ids[category_no]
