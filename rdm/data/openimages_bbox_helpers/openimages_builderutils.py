import io
import math
import warnings
from enum import Enum
from itertools import cycle
from typing import (Dict, List, Literal, NamedTuple, Optional, Tuple, Union,
                    overload)

import numpy as np
import torch
from PIL import Image as pil_img
from PIL import ImageColor
from PIL import ImageDraw as pil_img_draw
from PIL import ImageFont
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import CenterCrop, RandomCrop, RandomHorizontalFlip
from torchvision.transforms import functional as F
from torchvision.transforms.functional import crop as f_crop
from torchvision.transforms.functional import get_image_size as get_image_size

BoundingBox = Tuple[float, float, float, float]  # x0, y0, w, h

CropMethodType = Literal['none', 'random', 'center', 'intelligent', 'random-2d']

class Annotation(NamedTuple):
    area: float
    image_id: str
    bbox: BoundingBox
    category_no: int
    category_id: str
    id: Optional[int] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_group_of: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_occluded: Optional[bool] = None
    is_depiction: Optional[bool] = None
    is_inside: Optional[bool] = None
    segmentation: Optional[Dict] = None


class ImageDescription(NamedTuple):
    id: int
    file_name: str
    original_size: Tuple[int, int]  # w, h
    url: Optional[str] = None
    license: Optional[int] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None
    flickr_url: Optional[str] = None
    flickr_id: Optional[str] = None
    coco_id: Optional[str] = None


class Category(NamedTuple):
    id: str
    super_category: Optional[str]
    name: str


# source: seaborn, color palette tab10
color_palette = [(30, 118, 179), (255, 126, 13), (43, 159, 43), (213, 38, 39), (147, 102, 188),
                 (139, 85, 74), (226, 118, 193), (126, 126, 126), (187, 188, 33), (22, 189, 206)]
BLACK = (0, 0, 0)
GRAY_75 = (63, 63, 63)
GRAY_50 = (127, 127, 127)
GRAY_25 = (191, 191, 191)
WHITE = (255, 255, 255)
pil_to_tensor = transforms.PILToTensor()

GOLDEN_RATIO = (1. + math.sqrt(5)) / 2

def get_hue_value_from_class_id(class_id, n_classes):
    # n = class_id * GOLDEN_RATIO - math.floor(class_id * GOLDEN_RATIO)
    # hue = math.floor(n * 256)
    assert n_classes <= 360
    hue = int((float(class_id) / n_classes) * 360)
    return hue


def intersection_area(rectangle1, rectangle2):
    """
    Give intersection area of two rectangles.
    @param rectangle1: (x0, y0, w, h) of first rectangle
    @param rectangle2: (x0, y0, w, h) of second rectangle
    """
    rectangle1 = rectangle1[0], rectangle1[1], rectangle1[0] + rectangle1[2], rectangle1[1] + rectangle1[3]
    rectangle2 = rectangle2[0], rectangle2[1], rectangle2[0] + rectangle2[2], rectangle2[1] + rectangle2[3]
    x_overlap = max(0., min(rectangle1[2], rectangle2[2]) - max(rectangle1[0], rectangle2[0]))
    y_overlap = max(0., min(rectangle1[3], rectangle2[3]) - max(rectangle1[1], rectangle2[1]))
    return x_overlap * y_overlap


def draw_bboxes(image, bboxes,categories, n_categories,human_labels,fontsize=15):
    image = image.copy()
    width, height = image.size
    font = ImageFont.truetype('data/DejaVuSans.ttf', size=fontsize)
    for bbox, cls, readable in zip(bboxes, categories, human_labels):
        hue = get_hue_value_from_class_id(cls, n_categories)
        color = ImageColor.getrgb(f'hsv({hue}, 100%, 100%)')
        pil_bbox = [
            bbox[0] * width,
            bbox[1] * height,
            (bbox[0] + bbox[2]) * width,
            (bbox[1] + bbox[3]) * height
        ]
        center = [(pil_bbox[0]+pil_bbox[2]) //2,(pil_bbox[1]+pil_bbox[3]) //2]
        pil_img_draw.Draw(image).rectangle(pil_bbox, outline=color, width=3)
        pil_img_draw.Draw(image).text(center,readable,fill=color,font=font,align='center')
    return image


def draw_colorized_bboxes(size,bboxes, categories, n_categories, fill):
    background = pil_img.new('RGB',size)
    width, height = size
    for bbox, cls in zip(bboxes,categories):
        hue = get_hue_value_from_class_id(cls,n_categories)
        color = ImageColor.getrgb(f'hsv({hue}, 100%, 100%)')
        pil_bbox = [
            bbox[0] * width,
            bbox[1] * height,
            (bbox[0] + bbox[2]) * width,
            (bbox[1] + bbox[3]) * height
        ]
        pil_img_draw.Draw(background).rectangle(pil_bbox,outline=color,fill=color if fill else None,width=3)

    return background

def convert_pil_to_tensor(image):
    with warnings.catch_warnings():
        # to filter PyTorch UserWarning as described here: https://github.com/pytorch/vision/issues/2194
        warnings.simplefilter("ignore")
        return pil_to_tensor(image)


def figure_to_numpy(figure) -> np.ndarray:
    # adapted from https://stackoverflow.com/a/61443397/7952162
    io_buf = io.BytesIO()
    figure.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_array = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                           newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
    io_buf.close()
    img_array = (img_array / 255.).transpose((2, 0, 1))
    return img_array


def filter_annotations(annotations: List[Annotation], crop_coordinates: BoundingBox, crop_coordinates_min_area: float) \
        -> List:
    list_ = []
    for a in annotations:
        try:
            if intersection_area(a.bbox, crop_coordinates) / (a.bbox[2] * a.bbox[3]) >= crop_coordinates_min_area or \
                    intersection_area(a.bbox, crop_coordinates) / (crop_coordinates[2] * crop_coordinates[3]) >= 0.15:
                list_.append(a)
        except ZeroDivisionError:
            pass
    return list_


def additional_parameters_string(annotation: Annotation, short: bool = True) -> str:
    sl = slice(1) if short else slice(None)
    string = ''
    if not (annotation.is_group_of or annotation.is_occluded or annotation.is_depiction or annotation.is_inside):
        return string
    if annotation.is_group_of:
        string += 'group'[sl] + ','
    if annotation.is_occluded:
        string += 'occluded'[sl] + ','
    if annotation.is_depiction:
        string += 'depiction'[sl] + ','
    if annotation.is_inside:
        string += 'inside'[sl]
    return '(' + string.strip(",") + ')'


@overload
def clamp(x: int, minimum_value: int, maximum_value: int) -> int:
    ...


@overload
def clamp(x: float, minimum_value: float, maximum_value: float) -> float:
    ...


def clamp(x, minimum_value, maximum_value):
    return max(min(x, maximum_value), minimum_value)


class RandomCropReturnCoordinates(RandomCrop):
    def forward(self, img: Union[Tensor, Image]) -> (BoundingBox, Union[Tensor, Image]):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            Bounding box: x0, y0, w, h
            PIL Image or Tensor: Cropped image.

        Code source:
            torchvision.transforms.RandomCrop, torchvision 1.7.0
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        bbox = (j / width, i / height, w / width, h / height)  # x0, y0, w, h
        return bbox, F.crop(img, i, j, h, w)


class Random2dCropReturnCoordinates(torch.nn.Module):
    def __init__(self, min_size: int):
        super().__init__()
        self.min_size = min_size

    def forward(self, img: Union[Tensor, Image]) -> (BoundingBox, Union[Tensor, Image]):
        width, height = get_image_size(img)
        max_size = min(width, height)
        if max_size <= self.min_size:
            size = max_size
        else:
            size = int(np.random.randint(self.min_size, max_size+1))
        top = int(np.random.randint(0, height - size+1))
        left = int(np.random.randint(0, width - size+1))
        bbox = left / width, top / height, size / width, size / height
        return bbox, F.crop(img, top, left, size, size)


class CenterCropReturnCoordinates(CenterCrop):
    @staticmethod
    def get_bbox_of_center_crop(width: int, height: int) -> BoundingBox:
        if width > height:
            w = height / width
            h = 1.0
            x0 = 0.5 - w / 2
            y0 = 0.
        else:
            w = 1.0
            h = width / height
            x0 = 0.
            y0 = 0.5 - h / 2
        return x0, y0, w, h

    def forward(self, img: Union[Image, Tensor]) -> (BoundingBox, Union[Image, Tensor]):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            Bounding box: x0, y0, w, h
            PIL Image or Tensor: Cropped image.
        """
        width, height = get_image_size(img)
        return self.get_bbox_of_center_crop(width, height),  F.center_crop(img, self.size)


class IntelligentRandomCropReturnCoordinates(torch.nn.Module):
    def __init__(self, target_image_size: int, min_objects_per_image: int, max_objects_per_image: int,
                 crop_coordinates_min_area: float, bias_towards_more_objects: bool = False, no_samples: int = 100):
        super().__init__()
        self.target_image_size = target_image_size
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.crop_coordinates_min_area = crop_coordinates_min_area
        self.bias_towards_more_objects = bias_towards_more_objects
        self.no_samples = no_samples

    def _get_min_side_length(self, img: Union[Tensor, Image], annotations: List[Annotation]) -> int:
        """
        Get min side length in pixels
        """
        width, height = get_image_size(img)
        smallest_object_area_px_2 = min(a.bbox[2] * width * a.bbox[3] * height for a in annotations)
        min_side_length_px = int(np.sqrt(smallest_object_area_px_2 * self.crop_coordinates_min_area))
        min_side_length_px = max(min_side_length_px, self.target_image_size)
        return min_side_length_px

    @staticmethod
    def _get_max_side_length(img: Union[Tensor, Image]) -> int:
        """
        Get max side length in pixels
        """
        width, height = get_image_size(img)
        return min(width, height)

    def _make_bbox_candidates(self, img: Union[Tensor, Image], annotations: List[Annotation]) -> (List[Dict], Dict):
        width, height = get_image_size(img)
        min_side_length_px = self._get_min_side_length(img, annotations)
        max_side_length_px = self._get_max_side_length(img)
        if max_side_length_px < self.target_image_size or max_side_length_px < min_side_length_px:
            min_side_length_px = max_side_length_px

        area_sizes = np.random.randint(min_side_length_px ** 2, max_side_length_px ** 2 + 1, self.no_samples + 1)
        side_lengths = np.sqrt(area_sizes).astype(int).tolist()
        candidates = []
        fallback = None
        for side_length_px in side_lengths:
            x0_px = int(np.random.randint(0, width - side_length_px+1))
            y0_px = int(np.random.randint(0, height - side_length_px+1))
            bbox_candidate = (x0_px / width, y0_px / height, side_length_px / width, side_length_px / height)
            no_objects = len(filter_annotations(annotations, bbox_candidate, self.crop_coordinates_min_area))
            if not fallback:
                # first sample will always be fallback
                fallback = {'bbox': bbox_candidate, 'no_objects': no_objects}
            elif self.min_objects_per_image <= no_objects <= self.max_objects_per_image:
                # starting from the second, fill up candidates
                candidates.append({'bbox': bbox_candidate, 'no_objects': no_objects})

        if len(candidates) == 0:
            # happens in approx 0.5% of cases for COCO (after filtering out images with 0 or 1 object)
            return [fallback]
        return candidates

    def _pick_candidate(self, candidates: List[Dict]) -> Dict:
        possible_no_objects = list({c['no_objects'] for c in candidates})
        possible_no_objects.sort()
        candidates_by_no_objects = {
            no_objects: [c for c in candidates if c['no_objects'] == no_objects]
            for no_objects in possible_no_objects
        }
        weights = None
        if self.bias_towards_more_objects:
            weights = np.asarray([1.2 ** i for i, _ in enumerate(possible_no_objects)])
            # make prob
            weights /= weights.sum()

            raise NotImplementedError()
        no_objects = int(np.random.choice(possible_no_objects, p=weights))
        return int(np.random.choice(candidates_by_no_objects[no_objects]))

    def forward(self, img: Union[Tensor, Image], annotations: List[Annotation]) -> (BoundingBox, Union[Tensor, Image]):
        """
        Out of many possible random crops of the image, use the one that contains a number of objects within the given
        range. Even though a target image size is given (for minimum crop size), the image is not resized.
        Possibly bias probability towards a higher number of objects contained in the image.
        Args:
            img (PIL Image or Tensor): Image to be intelligently and randomly cropped.
            annotations: List of Annotations (with bounding boxes) of all relevant objects contained the image.
        Returns:
            Bounding box: x0, y0, w, h
            PIL Image or Tensor: Randomly cropped image.
        """
        candidates = self._make_bbox_candidates(img, annotations)
        candidate = self._pick_candidate(candidates)
        bbox = candidate['bbox']
        return bbox, crop_to_bbox(img, bbox)


class RandomHorizontalFlipReturnValue(RandomHorizontalFlip):
    def forward(self, img: Union[Tensor, Image]) -> (bool, Union[Tensor, Image]):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            flipped: whether the image was flipped or not
            PIL Image or Tensor: Randomly flipped image.

        Source:
            torchvision.transforms.RandomHorizontalFlip (version 1.7.0)
        """
        if torch.rand(1) < self.p:
            return True, F.hflip(img)
        return False, img


def crop_to_bbox(image: Union[Image, Tensor], bbox: BoundingBox) -> Union[Image, Tensor]:
    WIDTH, HEIGHT = get_image_size(image)
    left = clamp(int(bbox[0] * WIDTH), 0, WIDTH - 2)
    top = clamp(int(bbox[1] * HEIGHT), 0, HEIGHT - 2)
    width = clamp(int(bbox[2] * WIDTH), 1, WIDTH)
    height = clamp(int(bbox[3] * HEIGHT), 1, HEIGHT)
    crop = f_crop(image, top, left, height, width)
    if min(get_image_size(crop)) == 0:
        warnings.warn(f'Cropping failed for image.shape==[{image.shape}],  bbox=[{bbox}].')
        crop = image
    return crop

GraphRelationType = Literal['SixRelation']


class GraphSixRelation(Enum):
    surrounding = 0
    inside = 1
    above = 2
    below = 3
    right_of = 4
    left_of = 5

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.name)

    def horizontal_flip(self):
        if self.value <= 3:
            return self
        elif self is GraphSixRelation.right_of:
            return GraphSixRelation.left_of
        elif self is GraphSixRelation.left_of:
            return GraphSixRelation.right_of
        else:
            raise ValueError(f'No horizontal flip implemented for this element: [{self.name}]')


def get_bbox_six_relation(first: BoundingBox, second: BoundingBox) -> GraphSixRelation:
    if first == second:
        # raise ValueError('Cannot determine relation of two identical bounding boxes.')
        return int(np.random.choice([GraphSixRelation.surrounding, GraphSixRelation.inside]))  # not clean

    ax0, ay0, ax1, ay1 = first[0], first[1], first[0] + first[2], first[1] + first[3]
    bx0, by0, bx1, by1 = second[0], second[1], second[0] + second[2], second[1] + second[3]
    if ax0 < bx0 and ay0 < by0 and ax1 > bx1 and ay1 > by1:
        return GraphSixRelation.surrounding
    if ax0 > bx0 and ay0 > by0 and ax1 < bx1 and ay1 < by1:
        return GraphSixRelation.inside
    axc = (ax0 + ax1) / 2
    ayc = (ay0 + ay1) / 2
    bxc = (bx0 + bx1) / 2
    byc = (by0 + by1) / 2
    theta = math.atan2(ayc - byc, axc - bxc)
    if -math.pi / 4 <= theta < math.pi / 4:
        return GraphSixRelation.right_of
    if math.pi / 4 <= theta < 3 / 4 * math.pi:
        return GraphSixRelation.below
    if theta >= 3 / 4 * math.pi or theta <= -3 / 4 * math.pi:
        return GraphSixRelation.left_of
    if -3 / 4 * math.pi <= theta < -math.pi / 4:
        return GraphSixRelation.above
    raise RuntimeError('No valid relation could be determined.')


