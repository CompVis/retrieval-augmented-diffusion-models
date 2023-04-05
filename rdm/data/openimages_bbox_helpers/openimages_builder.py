import math
import random
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from more_itertools import take
from more_itertools.recipes import grouper
from PIL import Image as pil_image
from PIL import ImageDraw as pil_img_draw
from PIL import ImageFont
from torch import LongTensor, Tensor
from torch_geometric.data import Data

from rdm.data.openimages_bbox_helpers.openimages_builderutils import (
    BLACK, GRAY_75, WHITE, Annotation, BoundingBox, GraphRelationType,
    GraphSixRelation, additional_parameters_string, color_palette,
    convert_pil_to_tensor, figure_to_numpy, filter_annotations,
    get_bbox_six_relation)

try:    # Literal import compatible with all python versions
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


full_crop = (0., 0., 1., 1.)

def clamp(x: float):
    return max(min(x, 1.), 0.)

class DigraphSceneGraphBuilder:
    def build_digraph(self, *args, **kwargs) -> nx.DiGraph:
        raise NotImplementedError()


class SparseAsymmetricDigraphSceneGraphBuilder(DigraphSceneGraphBuilder):
    def __init__(self, no_object_classes: int, relation_type: GraphRelationType, crop_coordinates_min_area: float,
                 random_object_order: bool):
        self.no_object_classes = no_object_classes
        self.crop_coordinates_min_area = crop_coordinates_min_area
        self.random_object_order = random_object_order

        self.relation_type = relation_type
        if relation_type == 'SixRelation':
            self.no_relations = 6
            self.get_bbox_relation: Callable[[BoundingBox, BoundingBox], GraphSixRelation] = get_bbox_six_relation
        else:
            raise ValueError(f'Received invalid relation_type [{relation_type}].')

    @staticmethod
    def _add_node(graph: nx.DiGraph, node_id: int, annotation: Annotation) -> nx.DiGraph:
        graph.add_node(node_id, annotation=annotation, probability=1.)
        return graph

    def _add_edge(self, graph: nx.DiGraph, u: int, v: int, relation: Union[GraphSixRelation]) -> nx.DiGraph:
        if isinstance(relation, GraphSixRelation):
            graph.add_edge(u, v, relation_type=self.relation_type, relation=int(relation.value), probability=1.)
        else:
            raise ValueError(f'Received invalid `relation` of class [{type(relation).__name__}]')
        return graph

    def build_digraph(self, annotations: List[Annotation], crop_coordinates: Optional[BoundingBox] = None,
                      horizontal_flip: bool = False) -> nx.DiGraph:
        # possible todo: enforce connectedness of all nodes

        if len(annotations) == 0:
            raise ValueError('Received empty annotations list. Cannot build scene conditional_builder.')
        if crop_coordinates:
            filtered_annotations = filter_annotations(annotations, crop_coordinates, self.crop_coordinates_min_area)
            if len(filtered_annotations) == 0:
                # raise ValueError('After cropping to crop_coordinates, no annotations left.')
                pass
            else:
                annotations = filtered_annotations

        graph = nx.DiGraph()
        if self.random_object_order:
            np.random.shuffle(annotations)
        numbered_annotations: Dict[int, Annotation] = {i: ann for i, ann in enumerate(annotations)}
        for i, annotation in numbered_annotations.items():
            self._add_node(graph, i, annotation)
        if len(annotations) == 1:
            return graph

        for this_id, this in numbered_annotations.items():
            others = dict(numbered_annotations)
            others.pop(this_id)
            other_id = int(np.random.choice(list(others)))
            other = others[other_id]
            if graph.has_edge(this_id, other_id) or graph.has_edge(other_id, this_id):
                continue
            if np.random.sample() > 0.5:
                this, this_id, other, other_id = other, other_id, this, this_id
            relation = self.get_bbox_relation(this.bbox, other.bbox)
            if horizontal_flip:
                relation = relation.horizontal_flip()
            self._add_edge(graph, this_id, other_id, relation)

        return graph

    @staticmethod
    def plot(scene_graph: nx.DiGraph, label_for_category_no: Callable[[int], str], fig_size: Tuple[int, int],
             layout: Callable[[nx.DiGraph], Dict] = nx.circular_layout) -> Tensor:
        node_labels = {
            j: label_for_category_no(ann.category_no) + ' ' + additional_parameters_string(ann)
            for j, ann in scene_graph.nodes.data('annotation')
        }
        node_colors = take(len(node_labels), cycle(color_palette))
        node_colors = [(color[0] / 255, color[1] / 255, color[2] / 255) for color in node_colors]  # to float
        edge_labels = {(u, v): GraphSixRelation(relation).name for u, v, relation in scene_graph.edges.data('relation')}

        fig_size_dpi = (fig_size[0] / 100, fig_size[1] / 100)
        fig, ax = plt.subplots(1, 1, figsize=fig_size_dpi)
        layout_pos = layout(scene_graph)
        nx.draw_networkx(scene_graph, pos=layout_pos, ax=ax, labels=node_labels, node_color=node_colors, font_size=7)
        nx.draw_networkx_edge_labels(scene_graph, pos=layout_pos, ax=ax, edge_labels=edge_labels, font_size=7)
        plot_tensor = torch.from_numpy(figure_to_numpy(fig)[:3])
        plt.close()
        return plot_tensor

    def build(self, annotations: List[Annotation], crop_coordinates: Optional[BoundingBox] = None,
              horizontal_flip: bool = False) -> nx.DiGraph:
        return self.build_digraph(annotations, crop_coordinates, horizontal_flip)


def nx_graph_from_data(data: Data, edge_probability: Optional[List[float]] = None) -> nx.DiGraph:
    """
    @param data: torch_geometric data type with data.x, data.edge_index and data.edge_attr defined
    @param edge_probability: link probability with shape [no_edges]
    """
    graph = nx.DiGraph()
    if data.x is None:
        return graph
    for i, node in enumerate(data.x):
        category_no = node.argmax().item()
        category_probability = node[category_no].item() / node.sum().item()
        graph.add_node(i, category_no=category_no, probability=category_probability)
    for i, (edge, edge_attr) in enumerate(zip(data.edge_index.T, data.edge_attr.argmax(dim=1))):
        u, v = int(edge[0].item()), int(edge[1].item())
        relation = int(edge_attr.item())
        probability = edge_probability[i] if edge_probability is not None else None
        graph.add_edge(u, v, relation=relation, probability=probability)
    return graph



class TokenSceneGraphBuilder(SparseAsymmetricDigraphSceneGraphBuilder):
    def __init__(self, no_object_classes: int, relation_type: GraphRelationType, crop_coordinates_min_area: float,
                 random_object_order: bool, no_tokens: int, use_group_parameter: bool, use_additional_parameters: bool):
        super().__init__(no_object_classes, relation_type, crop_coordinates_min_area, random_object_order)
        self.no_tokens = no_tokens
        self.no_sections = int(math.sqrt(self.no_tokens))
        self.use_group_parameter = use_group_parameter
        self.use_additional_parameters = use_additional_parameters
        needed_tokens = no_object_classes + 2
        if use_group_parameter:
            needed_tokens = 2 * no_object_classes + 2
        if use_additional_parameters:
            needed_tokens = 16 * no_object_classes + 2
        if no_tokens < needed_tokens:
            raise ValueError(f'Not enough tokens [{no_tokens}] to represent all object classes [{no_object_classes}].')

    @property
    def separator(self) -> int:
        raise NotImplementedError()

    @property
    def none(self) -> int:
        raise NotImplementedError()

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError()

    def object_representation(self, annotation: Annotation) -> int:
        modifier = 0
        if self.use_group_parameter:
            modifier |= 1 * (annotation.is_group_of is True)
        if self.use_additional_parameters:
            modifier |= 2 * (annotation.is_occluded is True)
            modifier |= 4 * (annotation.is_depiction is True)
            modifier |= 8 * (annotation.is_inside is True)
        return annotation.category_no + self.no_object_classes * modifier

    def representation_to_annotation(self, representation: int) -> Annotation:
        category_no = representation % self.no_object_classes
        modifier = representation // self.no_object_classes
        # noinspection PyTypeChecker
        return Annotation(
            area=None,
            image_id=None,
            bbox=None,
            category_no=category_no,
            category_id=None,
            id=None,
            source=None,
            confidence=None,
            is_group_of=bool((modifier & 1) * self.use_group_parameter),
            is_occluded=bool((modifier & 2) * self.use_additional_parameters),
            is_depiction=bool((modifier & 4) * self.use_additional_parameters),
            is_inside=bool((modifier & 8) * self.use_additional_parameters)
        )

    def tokenize_coordinates(self, x: float, y: float) -> int:
        """
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        x_discrete = int(round(x * (self.no_sections - 1)))
        y_discrete = int(round(y * (self.no_sections - 1)))
        return y_discrete * self.no_sections + x_discrete

    def coordinates_from_token(self, token: int) -> (float, float):
        x = token % self.no_sections
        y = token // self.no_sections
        return x / (self.no_sections - 1), y / (self.no_sections - 1)

    @staticmethod
    def _rescale_annotations(annotations: List[Annotation], crop_coordinates: BoundingBox, flip: bool) -> \
            List[Annotation]:
        def rescale_bbox(bbox: BoundingBox) -> BoundingBox:
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            if flip:
                x0 = 1 - (x0 + w)
            return x0, y0, w, h

        return [a._replace(bbox=rescale_bbox(a.bbox)) for a in annotations]

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None,
              horizontal_flip: bool = False) -> LongTensor:
        raise NotImplementedError()


class ThreeTokenSceneGraphBuilder(TokenSceneGraphBuilder):
    def __init__(self, no_object_classes: int, relation_type: GraphRelationType, crop_coordinates_min_area: float,
                 no_max_relations: int, use_separator: bool, table_of_contents_type: int, random_object_order: bool,
                 no_tokens: int, use_group_parameter: bool, use_additional_parameters: bool):
        super().__init__(no_object_classes, relation_type, crop_coordinates_min_area, random_object_order, no_tokens,
                         use_group_parameter, use_additional_parameters)
        self.no_max_relations = no_max_relations
        self.use_separator = use_separator

        if table_of_contents_type not in [1, 2]:
            raise ValueError('Only values [1, 2] allowed for table_of_content')
        self.table_of_contents_type = table_of_contents_type

    @property
    def separator(self) -> int:
        return self.no_object_classes + self.no_relations + 1

    @property
    def none(self) -> int:
        return self.no_object_classes + self.no_relations + 2

    @property
    def relationship_entry_length(self) -> int:
        return 4 if self.use_separator else 3

    @property
    def toc_entry_length(self) -> int:
        return self.table_of_contents_type + int(self.use_separator)

    @property
    def embedding_dim(self) -> int:
        toc = 0
        if self.table_of_contents_type:
            toc = self.table_of_contents_type + int(self.use_separator)
        return self.no_max_relations * (toc + self.relationship_entry_length)

    @staticmethod
    def _pad(list_: List, pad_element: Any, pad_to_length: int) -> List:
        return list_ + [pad_element for _ in range(pad_to_length - len(list_))]

    def _make_table_of_contents(self, node_representations: List[int]) -> List[Tuple[int, ...]]:
        if self.table_of_contents_type == 0:
            return []
        elif self.table_of_contents_type == 1:
            table_of_contents = [(representation,) for representation in node_representations]
            empty_toc = (self.none,)
        else:
            table_of_contents = [(i, representation) for i, representation in enumerate(node_representations)]
            empty_toc = (self.none, self.none)
        table_of_contents = self._pad(table_of_contents, empty_toc, self.no_max_relations)
        if self.use_separator:
            table_of_contents = [tuple_ + (self.separator,) for tuple_ in table_of_contents]
        return table_of_contents

    def _make_relationships(self, digraph: nx.DiGraph) -> List[Tuple[int, ...]]:
        relationships: List[Tuple[int, ...]] = []
        for u, v, relation in digraph.edges.data('relation'):
            relationships.append((u, self.no_object_classes + relation, v))

        empty_relationship = tuple(self.none for _ in range(3))
        relationships = self._pad(relationships, empty_relationship, self.no_max_relations)
        if self.use_separator:
            relationships = [tuple_ + (self.separator,) for tuple_ in relationships]
        return relationships

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> LongTensor:
        digraph = self.build_digraph(annotations[:self.no_max_relations], crop_coordinates, horizontal_flip)
        if digraph.number_of_edges() > self.no_max_relations:
            raise RuntimeError(f'Received more relationships [{digraph.number_of_edges()}] '
                               f'than allowed [{self.no_max_relations}].')

        # building list from range adds a sanity check (continuous node numbers)
        object_representations = [
            self.object_representation(digraph.nodes[i]['annotation'])
            for i in range(digraph.number_of_nodes())
        ]
        table_of_contents = self._make_table_of_contents(object_representations)

        relationships = self._make_relationships(digraph)
        tupled_up_tokens = table_of_contents + relationships
        tokens = [token for tuple_ in tupled_up_tokens for token in tuple_]
        assert len(tokens) == self.embedding_dim
        return LongTensor(tokens)


class CoordinatesCenterPointsConditionalBuilder(TokenSceneGraphBuilder):
    def __init__(self, no_object_classes: int, relation_type: GraphRelationType, crop_coordinates_min_area: float,
                 no_max_objects: int, use_separator: bool, random_object_order: bool, no_tokens: int,
                 use_group_parameter: bool, use_additional_parameters: bool, encode_crop: bool):
        super().__init__(no_object_classes, relation_type, crop_coordinates_min_area, random_object_order, no_tokens,
                         use_group_parameter, use_additional_parameters)
        self.no_max_objects = no_max_objects
        self.use_separator = use_separator
        self.no_tokens = no_tokens
        self.encode_crop = encode_crop

    @property
    def separator(self):
        return self.no_tokens - 2

    @property
    def none(self):
        return self.no_tokens - 1

    @property
    def toc_entry_length(self):
        return 2 + int(self.use_separator)

    @property
    def embedding_dim(self):
        extra_length = 2 if self.encode_crop else 0
        return self.no_max_objects * self.toc_entry_length + extra_length

    @staticmethod
    def _pad(list_, pad_element, pad_to_length):
        return list_ + [pad_element for _ in range(pad_to_length - len(list_))]

    @staticmethod
    def _horizontally_flip_bbox(bbox: BoundingBox) -> BoundingBox:
        return 1 - (bbox[0] + bbox[2]), bbox[1], bbox[2], bbox[3]

    def _bbox_from_token_pair(self, token1: int, token2: int) -> BoundingBox:
        x0, y0 = self.coordinates_from_token(token1)
        x1, y1 = self.coordinates_from_token(token2)
        return x0, y0, x1 - x0, y1 - y0

    def _token_pair_from_bbox(self, bbox: BoundingBox) -> Tuple[int, int]:
        return self.tokenize_coordinates(bbox[0], bbox[1]), \
               self.tokenize_coordinates(bbox[0] + bbox[2], bbox[1] + bbox[3])

    def _make_table_of_contents(self, annotations: List[Annotation]) -> List[Tuple[int, ...]]:
        table_of_contents = [
            (self.object_representation(a),
             self.tokenize_coordinates(a.bbox[0] + a.bbox[2] / 2, a.bbox[1] + a.bbox[3] / 2))
            for a in annotations
        ]
        empty_toc = (self.none, self.none)
        table_of_contents = self._pad(table_of_contents, empty_toc, self.no_max_objects)
        if self.use_separator:
            table_of_contents = [tuple_ + (self.separator,) for tuple_ in table_of_contents]
        return table_of_contents

    def _crop_encoder(self, crop_coordinates: BoundingBox) -> List[int]:
        return list(self._token_pair_from_bbox(crop_coordinates))

    def inverse_build(self, conditional: LongTensor) \
            -> Tuple[List[Tuple[int, Tuple[float, float]]], Optional[BoundingBox]]:
        conditional_list = conditional.tolist()
        crop_coordinates = None
        if self.encode_crop:
            crop_coordinates = self._bbox_from_token_pair(conditional_list[-2], conditional_list[-1])
            conditional_list = conditional_list[:-2]
        table_of_content = grouper(conditional_list, self.toc_entry_length)
        assert conditional.shape[0] == self.embedding_dim
        return [
            (toc_entry[0], self.coordinates_from_token(toc_entry[1]))
            for toc_entry in table_of_content if toc_entry[0] != self.none
        ], crop_coordinates

    @staticmethod
    def get_plot_font_size(font_size: Optional[int], figure_size: Tuple[int, int]) -> int:
        if font_size is None:
            font_size = 10
            if max(figure_size) >= 256:
                font_size = 11
            if max(figure_size) >= 512:
                font_size = 14
        return font_size

    @staticmethod
    def get_circle_size(figure_size: Tuple[int, int]) -> int:
        circle_size = 2
        if max(figure_size) >= 256:
            circle_size = 3
        if max(figure_size) >= 512:
            circle_size = 4
        return circle_size

    @staticmethod
    def intify_bbox(bbox: BoundingBox, width: int, height: int) -> Tuple[int, int, int, int]:
        bbox = bbox[0] * width, bbox[1] * height, (bbox[0] + bbox[2]) * width, (bbox[1] + bbox[3]) * height
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    def plot(self, conditional: LongTensor, label_for_category_no: Callable[[int], str], figure_size: Tuple[int, int],
             line_width: int = 3, font_size: Optional[int] = None) -> Tensor:
        plot = pil_image.new('RGB', figure_size, WHITE)
        draw = pil_img_draw.Draw(plot)
        circle_size = self.get_circle_size(figure_size)
        font = ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
                                  size=self.get_plot_font_size(font_size, figure_size))
        width, height = plot.size
        description, crop_coordinates = self.inverse_build(conditional)
        for (representation, (x, y)), color in zip(description, cycle(color_palette)):
            x_abs, y_abs = x * width, y * height
            ann = self.representation_to_annotation(representation)
            label = label_for_category_no(ann.category_no) + ' ' + additional_parameters_string(ann)
            ellipse_bbox = [x_abs - circle_size, y_abs - circle_size, x_abs + circle_size, y_abs + circle_size]
            draw.ellipse(ellipse_bbox, fill=color, width=0)
            draw.text((x_abs, y_abs), label, anchor='md', fill=BLACK, font=font)
        if crop_coordinates is not None:
            draw.rectangle(self.intify_bbox(crop_coordinates, width, height), outline=GRAY_75, width=line_width)
        return convert_pil_to_tensor(plot) / 255

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> LongTensor:
        if len(annotations) == 0:
            raise ValueError('Did not receive any annotations.')
        if not crop_coordinates:
            crop_coordinates = full_crop

        if self.random_object_order:
            random.shuffle(annotations)

        if self.encode_crop:
            annotations = annotations[:self.no_max_objects]
            annotations = self._rescale_annotations(annotations, full_crop, horizontal_flip)
            if horizontal_flip:
                crop_coordinates = self._horizontally_flip_bbox(crop_coordinates)
            extra = self._crop_encoder(crop_coordinates)
        else:
            filtered_annotations = filter_annotations(annotations, crop_coordinates, self.crop_coordinates_min_area)
            if len(filtered_annotations) == 0:
                annotations = [annotations[0]]
            else:
                annotations = filtered_annotations[:self.no_max_objects]
            annotations = self._rescale_annotations(annotations, crop_coordinates, horizontal_flip)
            extra = []

        table_of_contents = self._make_table_of_contents(annotations)
        flattened = [token for tuple_ in table_of_contents for token in tuple_] + extra
        assert len(flattened) == self.embedding_dim
        assert all(0 <= value < self.no_tokens for value in flattened)
        return LongTensor(flattened)



class CoordinatesBoundingBoxConditionalBuilder(CoordinatesCenterPointsConditionalBuilder):

    @property
    def toc_entry_length(self) -> int:
        return 3 + int(self.use_separator)

    def _make_table_of_contents(self, annotations: List[Annotation]) -> List[Tuple[int, ...]]:
        table_of_contents = [
            (self.object_representation(ann), *self._token_pair_from_bbox(ann.bbox))
            for ann in annotations
        ]
        empty_toc = (self.none, self.none, self.none)
        table_of_contents = self._pad(table_of_contents, empty_toc, self.no_max_objects)
        if self.use_separator:
            table_of_contents = [tuple_ + (self.separator,) for tuple_ in table_of_contents]
        return table_of_contents

    def inverse_build(self, conditional: LongTensor) -> Tuple[List[Tuple[int, BoundingBox]], Optional[BoundingBox]]:
        conditional_list = conditional.tolist()
        crop_coordinates = None
        if self.encode_crop:
            crop_coordinates = self._bbox_from_token_pair(conditional_list[-2], conditional_list[-1])
            conditional_list = conditional_list[:-2]
        table_of_content = grouper(conditional_list, self.toc_entry_length)
        assert conditional.shape[0] == self.embedding_dim
        return [
            (toc_entry[0], self._bbox_from_token_pair(toc_entry[1], toc_entry[2]))
            for toc_entry in table_of_content if toc_entry[0] != self.none
        ], crop_coordinates

    def plot(self, conditional: LongTensor, label_for_category_no: Callable[[int], str], figure_size: Tuple[int, int],
             line_width: int = 3, font_size: Optional[int] = None) -> Tensor:
        plot = pil_image.new('RGB', figure_size, WHITE)
        draw = pil_img_draw.Draw(plot)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
            size=self.get_plot_font_size(font_size, figure_size)
        )
        width, height = plot.size
        description, crop_coordinates = self.inverse_build(conditional)
        for (representation, bbox), color in zip(description, cycle(color_palette)):
            annotation = self.representation_to_annotation(representation)
            class_label = label_for_category_no(annotation.category_no) + ' ' + additional_parameters_string(annotation)
            bbox = self.intify_bbox(bbox, width, height)
            draw.rectangle(bbox, outline=color, width=line_width)
            draw.text((bbox[0] + line_width, bbox[1] + line_width), class_label, anchor='la', fill=BLACK, font=font)
        if crop_coordinates is not None:
            draw.rectangle(self.intify_bbox(crop_coordinates, width, height), outline=GRAY_75, width=line_width)
        return convert_pil_to_tensor(plot) / 255


class TorchGeometricSceneGraphBuilder(SparseAsymmetricDigraphSceneGraphBuilder):
    @property
    def x_dim(self) -> int:
        raise NotImplementedError()

    @property
    def edge_attr_dim(self) -> Optional[int]:
        raise NotImplementedError()

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> Data:
        raise NotImplementedError()


class EdgeAttributeSceneGraphBuilder(TorchGeometricSceneGraphBuilder):
    @property
    def x_dim(self) -> int:
        return self.no_object_classes

    @property
    def edge_attr_dim(self) -> int:
        return self.no_relations

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> Data:
        digraph = self.build_digraph(annotations, crop_coordinates, horizontal_flip)

        category_nos = [ann.category_no for _, ann in digraph.nodes.data('annotation')]
        x = one_hot(torch.tensor(category_nos), self.x_dim)
        edge_index = LongTensor([[], []])
        edge_attr = LongTensor([[]] * self.edge_attr_dim).T
        if digraph.number_of_nodes() == 1:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        for u, v, relation in digraph.edges.data('relation'):
            if self.relation_type == 'SixRelation':
                new_edge = LongTensor([[u], [v]])
                edge_index = torch.cat((edge_index, new_edge), dim=1)
                new_edge_attribute = one_hot(LongTensor([relation]), self.edge_attr_dim)
                edge_attr = torch.cat((edge_attr, new_edge_attribute), dim=0)
            else:
                raise ValueError(f'Got invalid `relation_type` [{self.relation_type}]')
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class EdgeFeatureAsNodeSceneGraphBuilder(TorchGeometricSceneGraphBuilder):
    @property
    def x_dim(self) -> int:
        return self.no_object_classes + self.no_relations

    @property
    def edge_attr_dim(self) -> Optional[int]:
        return None

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> Data:
        digraph = self.build_digraph(annotations, crop_coordinates, horizontal_flip)

        category_nos = [ann.category_no for _, ann in digraph.nodes.data('annotation')]
        x = one_hot(torch.tensor(category_nos), self.x_dim)
        edge_index = LongTensor([[], []])
        if len(annotations) == 1:
            return Data(x=x, edge_index=edge_index)
        for u, v, relation in digraph.edges.data('relation'):
            if self.relation_type == 'SixRelation':
                new_x = one_hot(LongTensor([relation + self.no_object_classes]), self.x_dim)
                x = torch.cat((x, new_x), dim=0)
                via_node_id = x.shape[0] - 1
                new_edges = LongTensor([[u, via_node_id], [via_node_id, v]])
                edge_index = torch.cat((edge_index, new_edges), dim=1)
            else:
                raise ValueError(f'Got invalid `relation_type` [{self.relation_type}]')
        return Data(x=x, edge_index=edge_index)


def has_edge(edge_index: Tensor, u: int, v: int) -> bool:
    assert edge_index.shape[0] == 2
    edge = torch.tensor([[u], [v]])
    return (edge_index == edge).all(dim=0).any()


def out_edges_of_u(edge_index: Tensor, u: int) -> Tensor:
    relevant_edges = (edge_index[0] == u)
    return edge_index[1][relevant_edges].long()


def in_edges_of_v(edge_index: Tensor, v: int) -> Tensor:
    relevant_edges = (edge_index[1] == v)
    return edge_index[0][relevant_edges].long()


def efan_edges_have_edge(x: Tensor, edge_index: Tensor, u: int, v: int, edge_feature: Tensor) -> bool:
    u_out_edges = set(out_edges_of_u(edge_index, u).tolist())
    v_in_edges = set(in_edges_of_v(edge_index, v).tolist())
    via_nodes = u_out_edges & v_in_edges
    if not via_nodes:
        return False
    for via_node in via_nodes:
        if torch.equal(x[via_node].squeeze(), edge_feature.squeeze()):
            return True
    return False


def horizontally_flip_scene_graph(scene_graph: Data) -> Data:
    # only for SixRelation scene graphs
    edge_attr = scene_graph.edge_attr
    if edge_attr is not None:
        edge_attr = torch.index_select(edge_attr, 1, torch.LongTensor([0, 1, 3, 2, 5, 4]))
    return Data(x=scene_graph.x, edge_index=scene_graph.edge_index, edge_attr=edge_attr)

class RescaledAnnotationsBuilder(TokenSceneGraphBuilder):
    def __init__(self, no_object_classes: int, relation_type: GraphRelationType, crop_coordinates_min_area: float,
                 random_object_order: bool, no_tokens: int, use_group_parameter: bool, use_additional_parameters: bool,
                 no_max_objects: int):
        super().__init__(no_object_classes, relation_type, crop_coordinates_min_area, random_object_order, no_tokens,
                         use_group_parameter, use_additional_parameters)
        self.no_max_objects = no_max_objects

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> List[Annotation]:
        if len(annotations) == 0:
            raise ValueError('Did not receive any annotations.')
        if not crop_coordinates:
            crop_coordinates = (0., 0., 1., 1.)

        if self.random_object_order:
            random.shuffle(annotations)

        filtered_annotations = filter_annotations(annotations, crop_coordinates, self.crop_coordinates_min_area)
        if len(filtered_annotations) == 0:
            annotations = [annotations[0]]
        else:
            annotations = filtered_annotations[:self.no_max_objects]

        annotations = self._rescale_annotations(annotations, crop_coordinates, horizontal_flip)
        return annotations
