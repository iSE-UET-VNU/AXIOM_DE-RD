# Vendored from run-llama/ParseBench under Apache-2.0 (see NOTICE.md).
# Derived from the standalone kdl_frontier_nano.py provider.
import asyncio
import base64
import enum
import html
import io
import itertools
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import Counter
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, Dict, List, Literal, Match, Tuple, Union, cast
import httpx
import markdown as _markdown_lib
from PIL import Image
from pydantic import BaseModel, Field, computed_field, model_validator
logger = logging.getLogger('kdl_frontier_nano')

class NanoUsage:

    def __init__(self) -> None:
        self.stage_calls: Counter[str] = Counter()
        self.stage_sequences: Counter[str] = Counter()
        self.stage_latency_ms_sum: Counter[str] = Counter()
        self.stage_failures: Counter[str] = Counter()
        self.stage_retries: Counter[str] = Counter()
        self.batch_size_counts: Counter[str] = Counter()
        self.batch_fallback_sequences: Counter[str] = Counter()
        self.batch_fallback_recovered_sequences: Counter[str] = Counter()
        self.unrecovered_sequences: Counter[str] = Counter()

    def record(self, stage: str, *, sequences: int, latency_ms: float, failures: int=0, retries: int=0) -> None:
        self.stage_calls[stage] += 1
        self.stage_sequences[stage] += sequences
        self.stage_latency_ms_sum[stage] += latency_ms
        self.stage_failures[stage] += failures
        self.stage_retries[stage] += retries
        self.batch_size_counts[str(sequences)] += 1

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        return {'stage_calls': dict(self.stage_calls), 'stage_sequences': dict(self.stage_sequences), 'stage_latency_ms_sum': {key: round(float(value), 3) for key, value in self.stage_latency_ms_sum.items()}, 'stage_failures': dict(self.stage_failures), 'stage_retries': dict(self.stage_retries), 'batch_size_counts': dict(self.batch_size_counts), 'batch_fallback_sequences': dict(self.batch_fallback_sequences), 'batch_fallback_recovered_sequences': dict(self.batch_fallback_recovered_sequences), 'unrecovered_sequences': dict(self.unrecovered_sequences)}

class SequenceLimiter:

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError('SequenceLimiter capacity must be positive')
        self.capacity = capacity
        self._available = capacity
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def hold(self, sequences: int):
        weight = max(1, int(sequences))
        if weight > self.capacity:
            raise ValueError(f'Request needs {weight} sequences but capacity is {self.capacity}')
        async with self._condition:
            await self._condition.wait_for(lambda: self._available >= weight)
            self._available -= weight
        try:
            yield
        finally:
            async with self._condition:
                self._available += weight
                self._condition.notify_all()
Enum = enum.Enum
from typing import Literal as _Literal
RecognitionStage = _Literal['text', 'table', 'picture', 'formula']
RECOGNITION_STAGES: tuple = ('text', 'table', 'picture', 'formula')

class ElementCategory(str, Enum):
    TITLE = 'Title'
    SECTION_HEADER = 'Section-header'
    TEXT = 'Text'
    PAGE_HEADER = 'Page-header'
    PAGE_FOOTER = 'Page-footer'
    LIST_ITEM = 'List-item'
    CAPTION = 'Caption'
    FOOTNOTE = 'Footnote'
    TABLE = 'Table'
    PICTURE = 'Picture'
    DATA_CHART = 'Chart'
    FLOW_DIAGRAM = 'Flowchart'
    FORMULA = 'Formula'
'Recognition bucket contract shared by planning and layout grouping.'
from typing import Sequence
RecognitionBucket = RecognitionStage
RECOGNITION_BUCKETS: tuple[RecognitionBucket, ...] = RECOGNITION_STAGES
CATEGORY_TO_RECOGNITION_BUCKET: dict[str, RecognitionBucket] = {ElementCategory.TITLE.value: 'text', ElementCategory.SECTION_HEADER.value: 'text', ElementCategory.TEXT.value: 'text', ElementCategory.PAGE_HEADER.value: 'text', ElementCategory.PAGE_FOOTER.value: 'text', ElementCategory.LIST_ITEM.value: 'text', ElementCategory.CAPTION.value: 'text', ElementCategory.FOOTNOTE.value: 'text', ElementCategory.TABLE.value: 'table', ElementCategory.PICTURE.value: 'picture', ElementCategory.DATA_CHART.value: 'picture', ElementCategory.FLOW_DIAGRAM.value: 'picture', ElementCategory.FORMULA.value: 'formula'}

def category_to_recognition_bucket(category: str | ElementCategory) -> RecognitionBucket:
    return CATEGORY_TO_RECOGNITION_BUCKET.get(_category_value(category), 'text')

def target_categories_to_recognition_buckets(target_categories: Sequence[str] | None) -> frozenset[RecognitionBucket] | None:
    if not target_categories:
        return None
    return frozenset((CATEGORY_TO_RECOGNITION_BUCKET[category] for category in (_category_value(category) for category in target_categories) if category in CATEGORY_TO_RECOGNITION_BUCKET))

def _category_value(category: str | ElementCategory) -> str:
    if isinstance(category, ElementCategory):
        return category.value
    return str(category).strip()
'Layout category contract shared by layout provider adapters.'
from collections.abc import Mapping
from typing import Any
CANONICAL_LAYOUT_CATEGORIES = (ElementCategory.TITLE.value, ElementCategory.SECTION_HEADER.value, ElementCategory.TEXT.value, ElementCategory.LIST_ITEM.value, ElementCategory.TABLE.value, ElementCategory.PICTURE.value, ElementCategory.DATA_CHART.value, ElementCategory.FLOW_DIAGRAM.value, ElementCategory.FORMULA.value, ElementCategory.CAPTION.value, ElementCategory.FOOTNOTE.value, ElementCategory.PAGE_HEADER.value, ElementCategory.PAGE_FOOTER.value)
DEFAULT_LAYOUT_CATEGORY = ElementCategory.TEXT.value
_CANONICAL_BY_CASEFOLD = {category.casefold(): category for category in CANONICAL_LAYOUT_CATEGORIES}

def normalize_layout_category(raw_category: Any, provider_category_map: Mapping[str, str] | None=None) -> str:
    mapped_category = _map_provider_category(raw_category, provider_category_map)
    return _canonicalize_category(mapped_category)

def layout_recognition_bucket(category: Any) -> str:
    normalized_category = normalize_layout_category(category)
    return category_to_recognition_bucket(normalized_category)

def _map_provider_category(raw_category: Any, provider_category_map: Mapping[str, str] | None) -> str:
    category = _stringify_category(raw_category)
    if not provider_category_map:
        return category
    return provider_category_map.get(category) or provider_category_map.get(category.casefold()) or category

def _canonicalize_category(category: Any) -> str:
    canonical = _CANONICAL_BY_CASEFOLD.get(_stringify_category(category).casefold())
    if canonical is None:
        return DEFAULT_LAYOUT_CATEGORY
    return canonical

def _stringify_category(category: Any) -> str:
    if category is None:
        return ''
    return str(category).strip()
import math
from typing import Tuple
from PIL import Image
from pydantic import BaseModel, Field

class ImageConfig:
    FACTOR = 28
    MIN_PIXELS = 147384 * 2
    MAX_PIXELS = 2822400
    MAX_ASPECT_RATIO = 200
    DEFAULT_BACKGROUND_COLOR = (255, 255, 255)

class PageContentMetrics(BaseModel):
    foreground_ratio: float = Field(..., ge=0.0, le=1.0)
    edge_ratio: float = Field(..., ge=0.0, le=1.0)
    intensity_variance: float = Field(..., ge=0.0)
    is_blank: bool
CONTENT_ANALYSIS_MAX_DIMENSION = 192
FOREGROUND_DELTA_THRESHOLD = 12
FOREGROUND_DARK_THRESHOLD = 250
EDGE_DELTA_THRESHOLD = 18
BLANK_FOREGROUND_RATIO_THRESHOLD = 0.002
BLANK_EDGE_RATIO_THRESHOLD = 0.001
BLANK_INTENSITY_VARIANCE_THRESHOLD = 4.0

def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def normalize_image_mode(image: Image.Image, target_mode: str='RGB') -> Image.Image:
    if image.mode == target_mode:
        return image
    if image.mode == 'RGBA' and target_mode == 'RGB':
        background = Image.new('RGB', image.size, ImageConfig.DEFAULT_BACKGROUND_COLOR)
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert(target_mode)

def calculate_dimensions(width: int, height: int, factor: int=ImageConfig.FACTOR, min_pixels: int=ImageConfig.MIN_PIXELS, max_pixels: int=ImageConfig.MAX_PIXELS) -> Tuple[int, int]:
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > ImageConfig.MAX_ASPECT_RATIO:
        raise ValueError(f'종횡비가 너무 큽니다. 최대 {ImageConfig.MAX_ASPECT_RATIO}:1, 현재 {aspect_ratio:.1f}:1')
    target_height = max(factor, _round_by_factor(height, factor))
    target_width = max(factor, _round_by_factor(width, factor))
    if target_height * target_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        target_height = max(factor, _floor_by_factor(int(height / scale), factor))
        target_width = max(factor, _floor_by_factor(int(width / scale), factor))
    elif target_height * target_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        target_height = _ceil_by_factor(int(height * scale), factor)
        target_width = _ceil_by_factor(int(width * scale), factor)
        if target_height * target_width > max_pixels:
            scale = math.sqrt(target_height * target_width / max_pixels)
            target_height = max(factor, _floor_by_factor(int(target_height / scale), factor))
            target_width = max(factor, _floor_by_factor(int(target_width / scale), factor))
    return (target_width, target_height)

def smart_resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    try:
        target_width, target_height = calculate_dimensions(w, h)
        if (target_width, target_height) != (w, h):
            resized_img = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
            return resized_img
        else:
            return image
    except Exception:
        raise ValueError(f'Failed to resize image: {image}')

def is_monochromatic(image: Image.Image) -> bool:
    try:
        colors = image.getcolors()
        return colors is not None and len(colors) == 1
    except Exception as e:
        logger.warning(f'Failed to check monochromatic: {e}')
        return False

def analyze_page_content(image: Image.Image) -> PageContentMetrics:
    try:
        normalized = normalize_image_mode(image, 'RGB')
        width, height = normalized.size
        scale = min(1.0, CONTENT_ANALYSIS_MAX_DIMENSION / max(width, height))
        if scale < 1.0:
            sample_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            normalized = normalized.resize(sample_size, resample=Image.Resampling.BILINEAR)
        grayscale = normalized.convert('L')
        pixels = list(grayscale.getdata())
        total_pixels = len(pixels)
        if total_pixels == 0:
            return PageContentMetrics(foreground_ratio=0.0, edge_ratio=0.0, intensity_variance=0.0, is_blank=True)
        sorted_pixels = sorted(pixels)
        background_index = min(total_pixels - 1, int(total_pixels * 0.95))
        background_level = sorted_pixels[background_index]
        foreground_pixels = sum((1 for value in pixels if background_level - value >= FOREGROUND_DELTA_THRESHOLD and value < FOREGROUND_DARK_THRESHOLD))
        foreground_ratio = foreground_pixels / total_pixels
        mean = sum(pixels) / total_pixels
        intensity_variance = sum(((value - mean) ** 2 for value in pixels)) / total_pixels
        sample_width, sample_height = grayscale.size
        edge_pairs = 0
        edge_hits = 0
        for y in range(sample_height):
            row_offset = y * sample_width
            for x in range(sample_width):
                value = pixels[row_offset + x]
                if x + 1 < sample_width:
                    edge_pairs += 1
                    if abs(value - pixels[row_offset + x + 1]) >= EDGE_DELTA_THRESHOLD:
                        edge_hits += 1
                if y + 1 < sample_height:
                    edge_pairs += 1
                    if abs(value - pixels[row_offset + sample_width + x]) >= EDGE_DELTA_THRESHOLD:
                        edge_hits += 1
        edge_ratio = edge_hits / edge_pairs if edge_pairs else 0.0
        is_blank = foreground_ratio < BLANK_FOREGROUND_RATIO_THRESHOLD and edge_ratio < BLANK_EDGE_RATIO_THRESHOLD and (intensity_variance < BLANK_INTENSITY_VARIANCE_THRESHOLD)
        return PageContentMetrics(foreground_ratio=round(foreground_ratio, 6), edge_ratio=round(edge_ratio, 6), intensity_variance=round(float(intensity_variance), 6), is_blank=is_blank)
    except Exception as e:
        logger.warning(f'Failed to analyze page content: {e}')
        return PageContentMetrics(foreground_ratio=0.0, edge_ratio=0.0, intensity_variance=0.0, is_blank=False)

def preprocess_for_vlm(image: Image.Image) -> Image.Image:
    image = normalize_image_mode(image, 'RGB')
    image = smart_resize(image)
    return image
'the model layout token parsing utilities.'
import re
from typing import Any
from PIL import Image
from pydantic import BaseModel, Field
NATIVE_LAYOUT_IMAGE_SIZE = (1036, 1036)
_NATIVE_LAYOUT_RE = re.compile('<\\|box_start\\|>(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)<\\|box_end\\|><\\|ref_start\\|>([^<]+?)<\\|ref_end\\|>(?:(<\\|rotate_(?:up|right|down|left)\\|>))?', re.DOTALL)
_ROTATION_ANGLES = {'<|rotate_up|>': 0, '<|rotate_right|>': 90, '<|rotate_down|>': 180, '<|rotate_left|>': 270}
NATIVE_LAYOUT_CATEGORY_MAP = {'algorithm': 'Text', 'aside_text': 'Text', 'chart': 'Chart', 'code': 'Text', 'code_caption': 'Caption', 'equation': 'Formula', 'equation_block': 'Formula', 'footer': 'Page-footer', 'header': 'Page-header', 'image': 'Picture', 'image_block': 'Picture', 'image_caption': 'Caption', 'image_footnote': 'Footnote', 'inline_formula': 'Formula', 'list': 'List-item', 'list_item': 'List-item', 'page_footnote': 'Footnote', 'page_number': 'Page-footer', 'phonetic': 'Text', 'ref_text': 'List-item', 'table': 'Table', 'table_caption': 'Caption', 'table_footnote': 'Footnote', 'text': 'Text', 'title': 'Title', 'unknown': 'Text'}
_LIST_CHILD_CATEGORIES = {'text', 'list_item', 'ref_text'}
_IMAGE_BLOCK_CHILD_CATEGORIES = {'chart', 'image', 'image_caption', 'image_footnote'}
_EQUATION_CHILD_CATEGORIES = {'equation', 'inline_formula'}
_ATTACHMENT_PARENT_RAW_CATEGORIES = {'table_caption': {'table'}, 'table_footnote': {'table'}, 'image_caption': {'image', 'chart'}, 'image_footnote': {'image', 'chart'}, 'code_caption': {'code', 'algorithm'}}

class NativeLayoutItem(BaseModel):
    bbox: list[float]
    raw_bbox: list[int]
    raw_category: str
    layout_order: int
    angle: int = 0

class NormalizedNativeLayoutItem(BaseModel):
    bbox: list[float]
    category: str
    layout_order: int
    raw_category: str
    angle: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_output_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        if not data['metadata']:
            data.pop('metadata')
        return data

def is_native_layout_response(content: Any) -> bool:
    return isinstance(content, str) and '<|box_start|>' in content

def prepare_native_layout_image(image: Image.Image) -> Image.Image:
    return image.convert('RGB').resize(NATIVE_LAYOUT_IMAGE_SIZE, Image.Resampling.BICUBIC)

def parse_native_raw_layout_tokens(content: str) -> list[NativeLayoutItem]:
    items: list[NativeLayoutItem] = []
    for match in _NATIVE_LAYOUT_RE.finditer(content):
        converted_bbox = _convert_native_layout_bbox((match.group(1), match.group(2), match.group(3), match.group(4)))
        if converted_bbox is None:
            continue
        raw_bbox, bbox = converted_bbox
        raw_category = match.group(5).strip().lower()
        angle = _ROTATION_ANGLES.get(match.group(6) or '', 0)
        items.append(NativeLayoutItem(bbox=bbox, raw_bbox=raw_bbox, raw_category=raw_category, layout_order=len(items), angle=angle))
    return items

def parse_native_layout_tokens(content: str) -> list[dict[str, Any]]:
    raw_items = parse_native_raw_layout_tokens(content)
    normalized_items = normalize_native_layout_items(raw_items)
    return [item.to_output_dict() for item in normalized_items]

def normalize_native_layout_items(raw_items: list[NativeLayoutItem]) -> list[NormalizedNativeLayoutItem]:
    list_containers = [item for item in raw_items if item.raw_category == 'list' and _has_contained_child(item, raw_items, _LIST_CHILD_CATEGORIES)]
    image_blocks = [item for item in raw_items if item.raw_category == 'image_block' and _has_contained_child(item, raw_items, _IMAGE_BLOCK_CHILD_CATEGORIES)]
    equation_blocks = [item for item in raw_items if item.raw_category == 'equation_block']
    equation_child_orders = {child.layout_order for block in equation_blocks for child in raw_items if child.raw_category in _EQUATION_CHILD_CATEGORIES and _bbox_contains(block.bbox, child.bbox)}
    normalized_items: list[NormalizedNativeLayoutItem] = []
    for item in raw_items:
        metadata: dict[str, Any] = {}
        if item.raw_category == 'list' and item in list_containers:
            continue
        if item.raw_category == 'image_block' and item in image_blocks:
            continue
        if item.layout_order in equation_child_orders:
            continue
        if item.raw_category == 'equation_block':
            children = [_attachment_from_raw_item(child) for child in raw_items if child.raw_category in _EQUATION_CHILD_CATEGORIES and _bbox_contains(item.bbox, child.bbox)]
            if children:
                metadata['children'] = children
        else:
            list_parent = _find_containing_parent(item, list_containers)
            if list_parent is not None and item.raw_category in _LIST_CHILD_CATEGORIES:
                metadata.update(_parent_metadata(list_parent))
            image_parent = _find_containing_parent(item, image_blocks)
            if image_parent is not None:
                metadata.update(_parent_metadata(image_parent))
        normalized_items.append(NormalizedNativeLayoutItem(bbox=item.bbox, category=_category_for_item(item, metadata), layout_order=item.layout_order, raw_category=item.raw_category, angle=item.angle, metadata=metadata))
    _attach_caption_and_footnote_refs(normalized_items)
    return normalized_items

def _convert_native_layout_bbox(raw_bbox: tuple[str, str, str, str]) -> tuple[list[int], list[float]] | None:
    try:
        coords = [int(value) for value in raw_bbox]
    except ValueError:
        return None
    if any((coord < 0 or coord > 1000 for coord in coords)):
        return None
    x1, y1, x2, y2 = coords
    if x2 < x1:
        x1, x2 = (x2, x1)
    if y2 < y1:
        y1, y2 = (y2, y1)
    if x1 == x2 or y1 == y2:
        return None
    raw_coords = [x1, y1, x2, y2]
    normalized_coords = [round(value / 1000.0, 4) for value in raw_coords]
    return (raw_coords, normalized_coords)

def _category_for_item(item: NativeLayoutItem, metadata: dict[str, Any]) -> str:
    if item.raw_category in _LIST_CHILD_CATEGORIES and (item.raw_category in {'list_item', 'ref_text'} or metadata.get('parent_raw_category') == 'list'):
        return 'List-item'
    return normalize_layout_category(item.raw_category, NATIVE_LAYOUT_CATEGORY_MAP)

def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

def _bbox_contains(parent_bbox: list[float], child_bbox: list[float], *, min_child_overlap: float=0.85) -> bool:
    child_area = _bbox_area(child_bbox)
    if child_area <= 0:
        return False
    inter_x1 = max(parent_bbox[0], child_bbox[0])
    inter_y1 = max(parent_bbox[1], child_bbox[1])
    inter_x2 = min(parent_bbox[2], child_bbox[2])
    inter_y2 = min(parent_bbox[3], child_bbox[3])
    intersection = _bbox_area([inter_x1, inter_y1, inter_x2, inter_y2])
    return intersection / child_area >= min_child_overlap

def _has_contained_child(parent: NativeLayoutItem, items: list[NativeLayoutItem], child_categories: set[str]) -> bool:
    return any((child.layout_order != parent.layout_order and child.raw_category in child_categories and _bbox_contains(parent.bbox, child.bbox) for child in items))

def _find_containing_parent(item: NativeLayoutItem, parents: list[NativeLayoutItem]) -> NativeLayoutItem | None:
    candidates = [parent for parent in parents if parent.layout_order != item.layout_order and _bbox_contains(parent.bbox, item.bbox)]
    return min(candidates, key=lambda parent: _bbox_area(parent.bbox), default=None)

def _parent_metadata(parent: NativeLayoutItem) -> dict[str, Any]:
    return {'parent_raw_category': parent.raw_category, 'parent_bbox': parent.bbox, 'parent_layout_order': parent.layout_order}

def _attachment_from_raw_item(item: NativeLayoutItem) -> dict[str, Any]:
    return {'bbox': item.bbox, 'raw_category': item.raw_category, 'layout_order': item.layout_order, 'angle': item.angle}

def _attachment_from_normalized_item(item: NormalizedNativeLayoutItem) -> dict[str, Any]:
    return {'bbox': item.bbox, 'raw_category': item.raw_category, 'layout_order': item.layout_order, 'angle': item.angle}

def _attach_caption_and_footnote_refs(items: list[NormalizedNativeLayoutItem]) -> None:
    for item in items:
        parent_raw_categories = _ATTACHMENT_PARENT_RAW_CATEGORIES.get(item.raw_category)
        if not parent_raw_categories:
            continue
        parent = _find_nearest_attachment_parent(item, items, parent_raw_categories)
        if parent is None:
            continue
        key = 'attached_footnotes' if item.raw_category.endswith('_footnote') else 'attached_captions'
        parent.metadata.setdefault(key, []).append(_attachment_from_normalized_item(item))
        item.metadata.update({'attached_to_raw_category': parent.raw_category, 'attached_to_category': parent.category, 'attached_to_bbox': parent.bbox, 'attached_to_layout_order': parent.layout_order})

def _find_nearest_attachment_parent(item: NormalizedNativeLayoutItem, items: list[NormalizedNativeLayoutItem], parent_raw_categories: set[str]) -> NormalizedNativeLayoutItem | None:
    candidates = [candidate for candidate in items if candidate.layout_order != item.layout_order and candidate.raw_category in parent_raw_categories]
    return min(candidates, key=lambda candidate: _attachment_distance(candidate, item), default=None)

def _attachment_distance(parent: NormalizedNativeLayoutItem, child: NormalizedNativeLayoutItem) -> float:
    parent_center_x = (parent.bbox[0] + parent.bbox[2]) / 2
    child_center_x = (child.bbox[0] + child.bbox[2]) / 2
    horizontal_gap = abs(parent_center_x - child_center_x)
    if child.bbox[1] >= parent.bbox[3]:
        vertical_gap = child.bbox[1] - parent.bbox[3]
    elif parent.bbox[1] >= child.bbox[3]:
        vertical_gap = parent.bbox[1] - child.bbox[3]
    else:
        vertical_gap = 0.0
    return vertical_gap + horizontal_gap
import html
import itertools
import re
from typing import Any, Dict, List, Tuple, cast
from pydantic import BaseModel, computed_field, model_validator

class TableCell(BaseModel):
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode='before')
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        if isinstance(data, Dict):
            if 'text' in data:
                return data
            text = data['bbox'].get('token', '')
            if not len(text):
                text_cells = data.pop('text_cell_bboxes', None)
                if text_cells:
                    for el in text_cells:
                        text += el['token'] + ' '
                text = text.strip()
            data['text'] = text
        return data

class TableData(BaseModel):
    table_cells: List[TableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field
    def grid(self) -> List[List[TableCell]]:
        table_data = [[TableCell(text='', start_row_offset_idx=i, end_row_offset_idx=i + 1, start_col_offset_idx=j, end_col_offset_idx=j + 1) for j in range(self.num_cols)] for i in range(self.num_rows)]
        for cell in self.table_cells:
            for i in range(min(cell.start_row_offset_idx, self.num_rows), min(cell.end_row_offset_idx, self.num_rows)):
                for j in range(min(cell.start_col_offset_idx, self.num_cols), min(cell.end_col_offset_idx, self.num_cols)):
                    table_data[i][j] = cell
        return table_data
OTSL_NL = '<nl>'
OTSL_FCEL = '<fcel>'
OTSL_ECEL = '<ecel>'
OTSL_LCEL = '<lcel>'
OTSL_UCEL = '<ucel>'
OTSL_XCEL = '<xcel>'
NON_CAPTURING_TAG_GROUP = '(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)'
OTSL_FIND_PATTERN = re.compile(f'{NON_CAPTURING_TAG_GROUP}.*?(?={NON_CAPTURING_TAG_GROUP}|$)', flags=re.DOTALL)
IGNORABLE_OTSL_TAIL_PATTERN = re.compile('(?:\\s|<br\\s*/?>|<nl>|</otsl>)+', flags=re.IGNORECASE)

def otsl_extract_tokens_and_text(s: str) -> Tuple[List[str], List[str]]:
    pattern = '(' + '|'.join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + ')'
    tokens = re.findall(pattern, s)
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts if token.strip()]
    return (tokens, text_parts)

def otsl_parse_texts(texts: List[str], tokens: List[str]) -> Tuple[List[TableCell], List[List[str]]]:
    split_word = OTSL_NL
    split_row_tokens = [list(y) for x, y in itertools.groupby(tokens, lambda z: z == split_word) if not x]
    table_cells: List[TableCell] = []
    r_idx = 0
    c_idx = 0
    if split_row_tokens:
        max_cols = max((len(row) for row in split_row_tokens))
        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        new_texts = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    def count_right(tokens: List[List[str]], c_idx: int, r_idx: int, which_tokens: List[str]) -> int:
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens: List[List[str]], c_idx: int, r_idx: int, which_tokens: List[str]) -> int:
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span
    for i, text in enumerate(texts):
        cell_text = ''
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2
            next_right_cell = texts[i + right_offset] if i + right_offset < len(texts) else ''
            next_bottom_cell = ''
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]
            if next_right_cell in [OTSL_LCEL, OTSL_XCEL]:
                col_span += count_right(split_row_tokens, c_idx + 1, r_idx, [OTSL_LCEL, OTSL_XCEL])
            if next_bottom_cell in [OTSL_UCEL, OTSL_XCEL]:
                row_span += count_down(split_row_tokens, c_idx, r_idx + 1, [OTSL_UCEL, OTSL_XCEL])
            table_cells.append(TableCell(text=cell_text.strip(), row_span=row_span, col_span=col_span, start_row_offset_idx=r_idx, end_row_offset_idx=r_idx + row_span, start_col_offset_idx=c_idx, end_col_offset_idx=c_idx + col_span))
        if text in [OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return (table_cells, split_row_tokens)

def export_to_html(table_data: TableData) -> str:
    nrows = table_data.num_rows
    ncols = table_data.num_cols
    if len(table_data.table_cells) == 0:
        return ''
    body = ''
    grid = table_data.grid() if callable(table_data.grid) else table_data.grid
    for i in range(nrows):
        body += '<tr>'
        for j in range(ncols):
            cell: TableCell = grid[i][j]
            rowspan, rowstart = (cell.row_span, cell.start_row_offset_idx)
            colspan, colstart = (cell.col_span, cell.start_col_offset_idx)
            if rowstart != i or colstart != j:
                continue
            content = html.escape(cell.text.strip()).replace('\r\n', '\n')
            content = content.replace('\r', '\n').replace('\n', '<br>')
            celltag = 'th' if cell.column_header else 'td'
            opening_tag = f'{celltag}'
            if rowspan > 1:
                opening_tag += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening_tag += f' colspan="{colspan}"'
            body += f'<{opening_tag}>{content}</{celltag}>'
        body += '</tr>'
    body = f'<table>{body}</table>'
    return body

def otsl_pad_to_sqr_v2(otsl_str: str) -> str:
    assert isinstance(otsl_str, str)
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL
    lines = otsl_str.split(OTSL_NL)
    row_data: List[Dict[str, Any]] = []
    for line in lines:
        if not line:
            continue
        raw_cells = OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        total_len = len(raw_cells)
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1
        row_data.append({'raw_cells': raw_cells, 'total_len': total_len, 'min_len': min_len})
    if not row_data:
        return OTSL_NL
    global_min_width = max((row['min_len'] for row in row_data)) if row_data else 0
    max_total_len = max((row['total_len'] for row in row_data)) if row_data else 0
    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)
    min_total_cost = float('inf')
    optimal_width = search_end
    for width in range(search_start, search_end + 1):
        current_total_cost = sum((abs(row['total_len'] - width) for row in row_data))
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            optimal_width = width
    repaired_lines = []
    for row in row_data:
        cells = cast(List[str], row['raw_cells'])
        current_len = len(cells)
        if current_len > optimal_width:
            new_cells = cells
        else:
            padding = [OTSL_ECEL] * (optimal_width - current_len)
            new_cells = cells + padding
        repaired_lines.append(''.join(new_cells))
    return OTSL_NL.join(repaired_lines) + OTSL_NL

def convert_otsl_to_html(otsl_content: str) -> str:
    otsl_content = otsl_pad_to_sqr_v2(otsl_content)
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)
    table_data = TableData(num_rows=len(split_row_tokens), num_cols=max((len(row) for row in split_row_tokens)) if split_row_tokens else 0, table_cells=table_cells)
    return export_to_html(table_data)

def _collapse_placeholder_with_ignorable_tail(otsl_content: str, placeholders: List[str]) -> str:
    stripped = otsl_content.strip()
    for placeholder in placeholders:
        if stripped == placeholder:
            return placeholder
        if stripped.startswith(placeholder):
            tail = stripped[len(placeholder):].strip()
            if tail and IGNORABLE_OTSL_TAIL_PATTERN.fullmatch(tail):
                return placeholder
    return otsl_content
_OTSL_STRUCTURAL_TAG_RE = re.compile('<(?:otsl|fcel|ecel|lcel|ucel|xcel)>')

def _flatten_top_level_table_placeholders(otsl_content: str, placeholder_to_html: Dict[str, str]) -> str | None:
    present = [p for p in placeholder_to_html if p in otsl_content]
    if not present:
        return None
    remainder = otsl_content
    for placeholder in present:
        remainder = remainder.replace(placeholder, '')
    if _OTSL_STRUCTURAL_TAG_RE.search(remainder):
        return None
    ordered = sorted(present, key=otsl_content.index)
    return ''.join((placeholder_to_html[p] for p in ordered))

def convert_otsl_to_html_v2(otsl_content: str, debug: bool=False) -> str:
    import uuid
    if debug:
        logger.debug('OTSL conversion started', input_preview=otsl_content[:200])
    n_open = otsl_content.count('<otsl>')
    if n_open <= 1 and otsl_content.count('</otsl>') > n_open:
        flat = otsl_content.replace('<otsl>', '').replace('</otsl>', '')
        try:
            return convert_otsl_to_html(flat)
        except Exception as e:
            if debug:
                logger.error('Bug-G flat conversion failed', error=str(e))
            return '<table></table>'
    placeholder_to_otsl: Dict[str, str] = {}
    nested_pattern = '<otsl>((?:(?!<otsl>).)*?)</otsl>'
    iteration = 0
    max_iterations = 100
    while iteration < max_iterations:
        match = re.search(nested_pattern, otsl_content, re.DOTALL)
        if not match:
            break
        inner_otsl = match.group(1)
        if debug:
            logger.debug('Found nested OTSL', iteration=iteration, nested_preview=inner_otsl[:100])
        placeholder = f'__NESTED_TABLE_{uuid.uuid4().hex}__'
        placeholder_to_otsl[placeholder] = inner_otsl
        otsl_content = otsl_content[:match.start()] + placeholder + otsl_content[match.end():]
        iteration += 1
    if debug:
        logger.debug('After all nested replacement', content_preview=otsl_content[:200])
        logger.debug('Total nested levels found', total_levels=len(placeholder_to_otsl))
    otsl_content = re.sub('^<otsl>|</otsl>$', '', otsl_content.strip())
    otsl_content = _collapse_placeholder_with_ignorable_tail(otsl_content, list(placeholder_to_otsl))
    placeholder_to_html: Dict[str, str] = {}
    for placeholder, otsl in placeholder_to_otsl.items():
        try:
            inner_html = convert_otsl_to_html(otsl)
            placeholder_to_html[placeholder] = inner_html
            if debug:
                logger.debug(f'Converted {placeholder[:30]}... to HTML: {inner_html[:80]}...')
        except Exception as e:
            placeholder_to_html[placeholder] = '<table></table>'
            if debug:
                logger.error('OTSL conversion failed', placeholder=placeholder, error=str(e))
    top_stripped = otsl_content.strip()
    flat_html = _flatten_top_level_table_placeholders(top_stripped, placeholder_to_html)
    if top_stripped in placeholder_to_otsl:
        html_result = placeholder_to_html.get(top_stripped, '<table></table>')
    elif flat_html is not None:
        html_result = flat_html
    else:
        try:
            html_result = convert_otsl_to_html(otsl_content)
            if debug:
                logger.debug('Top-level HTML generated', html_preview=html_result[:150])
        except Exception as e:
            if debug:
                logger.error('Top-level conversion failed', error=str(e))
            return '<table></table>'
    max_replace_iterations = 10
    for _ in range(max_replace_iterations):
        replaced = False
        for placeholder, inner_html in placeholder_to_html.items():
            escaped_placeholder = html.escape(placeholder)
            if placeholder in html_result or escaped_placeholder in html_result:
                html_result = html_result.replace(placeholder, inner_html)
                html_result = html_result.replace(escaped_placeholder, inner_html)
                replaced = True
        if not replaced:
            break
    if debug:
        logger.debug('OTSL conversion completed', result_preview=html_result[:200])
    return str(html_result)
if __name__ == '__main__':
    simple_otsl = '<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>'
    print('=== Simple table (v1) ===')
    print(convert_otsl_to_html(simple_otsl))
    print()
    nested_otsl = '<otsl><fcel><otsl><fcel>연 번<fcel>품명<lcel><fcel>규격<fcel>신고인<fcel>증정국(인)<fcel>선물<br>가액<nl><ucel><fcel>사진<fcel>품목<ucel><ucel><ucel><ucel><nl><fcel>1<ecel><ecel><ecel><ecel><ecel><ecel><nl><fcel>(작성<br>예시)<fcel><otsl><fcel>\U000f012b<nl></otsl><fcel>다기<fcel>24cm×26cm❙<fcel>김동연<br>(도지사)<fcel>중국<br>(천리)<fcel>추정가<br>(약 15만원<br>또는<br>가액 추정 불가<nl></otsl><lcel><lcel><nl><fcel>장<fcel>관<fcel>항 목<nl><fcel>200세외수입<lcel><lcel><nl><ecel><fcel>210경상적 세외수입<lcel><nl><ecel><fcel>220임시적 세외수입<lcel><nl><ecel><ecel><fcel>221재산매각수입<nl><ucel><ecel><fcel>222자치단체간 부담금<nl><ucel><ecel><fcel>223보조금 반환수입<nl><ucel><ecel><fcel>224기타수입<nl><ucel><ecel><fcel>225지난년도 수입<nl><ucel><ecel><fcel>230지방행정제재부과금등<nl><ucel><fcel>240지난년도 수입<lcel><nl><ucel><ecel><fcel>241지난년도 수입<nl><ucel><ecel><fcel>241-01지난년도 수입<nl></otsl>'
    print('=== Nested table (v2) ===')
    print(convert_otsl_to_html_v2(nested_otsl))
    print()
    simple_nested = '<otsl><fcel>Outer A<fcel><otsl><fcel>Inner 1<fcel>Inner 2<nl></otsl><nl><fcel>Outer B<fcel>Outer C<nl></otsl>'
    print('=== Simple nested (v2) ===')
    print(convert_otsl_to_html_v2(simple_nested, debug=True))
import re
import unicodedata
from collections import Counter
from typing import Tuple, Union

def find_shortest_repeating_substring(s: str) -> Union[str, None]:
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                return substring
    return None

def find_repeating_suffix(s: str, min_len: int=8, min_repeats: int=5) -> Union[Tuple[str, str, int], None]:
    for i in range(len(s) // min_repeats, min_len - 1, -1):
        unit = s[-i:]
        if s.endswith(unit * min_repeats):
            count = 0
            temp_s = s
            while temp_s.endswith(unit):
                temp_s = temp_s[:-i]
                count += 1
            start_index = len(s) - count * i
            return (s[:start_index], unit, count)
    return None

def remove_ngram_repeats(text: str, ngram_size: int=2, min_repeats: int=3) -> str:
    if not text:
        return text
    words = text.split()
    if len(words) < ngram_size * min_repeats:
        return text
    i = 0
    result_words = []
    while i < len(words):
        if i + ngram_size > len(words):
            result_words.extend(words[i:])
            break
        current_ngram = tuple(words[i:i + ngram_size])
        repeat_count = 1
        j = i + ngram_size
        while j + ngram_size <= len(words):
            next_ngram = tuple(words[j:j + ngram_size])
            if next_ngram == current_ngram:
                repeat_count += 1
                j += ngram_size
            else:
                break
        if repeat_count >= min_repeats:
            result_words.extend(words[i:i + ngram_size])
            i = j
        else:
            result_words.append(words[i])
            i += 1
    return ' '.join(result_words)

def truncate_repetitive_content(content: str, line_threshold: int=10, char_threshold: int=10, min_len: int=10, preserve_line_breaks: bool=False) -> str:
    if preserve_line_breaks:
        lines = content.split('\n')
        processed_lines = []
        for line in lines:
            if not line.strip():
                processed_lines.append(line)
            else:
                processed = truncate_repetitive_content(line, line_threshold, char_threshold, min_len, preserve_line_breaks=False)
                processed_lines.append(processed)
        return '\n'.join(processed_lines)
    stripped_content = content.strip()
    if ' ' in stripped_content and len(stripped_content.split()) >= 4:
        words = stripped_content.split()
        if len(set(words)) == 1:
            content = words[0]
            stripped_content = content.strip()
    if ' ' in stripped_content and len(stripped_content.split()) >= 6:

        def _dedup_ngrams(line: str) -> str:
            line_3gram = remove_ngram_repeats(line, ngram_size=3, min_repeats=3)
            line_2gram = remove_ngram_repeats(line, ngram_size=2, min_repeats=3)
            candidate = line_3gram if len(line_3gram) < len(line_2gram) else line_2gram
            normalized = ' '.join(line.split())
            return candidate if len(candidate) < len(normalized) else line
        content = '\n'.join((_dedup_ngrams(line) if line.strip() else line for line in content.split('\n')))
        stripped_content = content.strip()

    def is_separator_symbol(char: str) -> bool:
        cat = unicodedata.category(char)
        return cat.startswith('P') or cat == 'Zs' or cat.startswith('S')
    match = re.search('(.)\\1{4,}', stripped_content)
    if match and is_separator_symbol(match.group(1)):
        repeated_char = match.group(1)
        content = re.sub(re.escape(repeated_char) + '{5,}', '', content)
        stripped_content = content.strip()
    if not stripped_content:
        return content
    if '\n' not in stripped_content and len(stripped_content) > 100:
        suffix_match = find_repeating_suffix(stripped_content, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, repeating_unit, count = suffix_match
            if len(repeating_unit) * count > len(stripped_content) * 0.5:
                return prefix
    if '\n' not in stripped_content and len(stripped_content) > min_len:
        repeating_substring = find_shortest_repeating_substring(stripped_content)
        if repeating_substring:
            count = len(stripped_content) // len(repeating_substring)
            if count >= char_threshold:
                return repeating_substring
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return content
    total_lines = len(lines)
    if total_lines < line_threshold:
        return content
    line_counts = Counter(lines)
    most_common_line, count = line_counts.most_common(1)[0]
    if count >= line_threshold and count / total_lines >= 0.8:
        return most_common_line
    return content
'\nPicture Response Validator\n\nVLM에서 반환된 picture/chart/flowchart 응답의 유효성을 검증합니다.\n'
import re
from decimal import Decimal
_SEPARATOR_PATTERN = re.compile('^\\s*\\|(?:\\s*:?-+:?\\s*\\|)+\\s*$')
_KRW_EXPR_PATTERN = re.compile('(?P<expr>(?:\\d[\\d,]*\\s*(?:조원|억원|만원|조|억|만|원)\\s*)+)')
_KRW_SEGMENT_PATTERN = re.compile('(\\d[\\d,]*)\\s*(조원|억원|만원|조|억|만|원)')
_KRW_UNIT_MULTIPLIERS = {'조원': 10 ** 12, '억원': 10 ** 8, '만원': 10 ** 4, '조': 10 ** 12, '억': 10 ** 8, '만': 10 ** 4, '원': 1}
_YEAR_HEADER_PATTERN = re.compile('^(year|년도|연도|年)$', re.IGNORECASE)
_AMOUNT_HEADER_PATTERN = re.compile('(amount|value|금액|예산|budget|won|krw|억원|억\\s*원|億ウォン|兆ウォン|trillion|billion|million)', re.IGNORECASE)
_PERCENT_PATTERN = re.compile('^\\d[\\d,]*(?:\\.\\d+)?%$')
_PLAIN_NUMBER_PATTERN = re.compile('^\\d[\\d,]*(?:\\.\\d+)?$')
_KRW_VALUE_PATTERN = re.compile('^KRW [\\d,]+$')

def split_markdown_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith('|'):
        row = row[1:]
    if row.endswith('|'):
        row = row[:-1]
    return [cell.strip() for cell in row.split('|')]

def normalize_markdown_table_content(content: str) -> str:
    if not content or not isinstance(content, str):
        return content
    normalized = content.replace('｜', '|')
    lines = normalized.splitlines()
    table_line_indices = [idx for idx, line in enumerate(lines) if line.strip().startswith('|')]
    if len(table_line_indices) < 2:
        return normalized
    first_idx = table_line_indices[0]
    header_line = lines[first_idx].strip()
    if first_idx + 1 < len(lines) and _SEPARATOR_PATTERN.match(lines[first_idx + 1].strip()):
        return normalized
    header_cells = split_markdown_table_row(header_line)
    if not header_cells:
        return normalized
    separator_line = '|' + '|'.join([' --- '] * len(header_cells)) + '|'
    lines.insert(first_idx + 1, separator_line)
    return '\n'.join(lines)

def normalize_krw_expressions(content: str) -> str:
    if not content or not isinstance(content, str):
        return content

    def _replace(match: re.Match[str]) -> str:
        expr = match.group('expr')
        total = 0
        found = False
        for raw_number, unit in _KRW_SEGMENT_PATTERN.findall(expr):
            found = True
            total += int(raw_number.replace(',', '')) * _KRW_UNIT_MULTIPLIERS[unit]
        if not found:
            return expr
        return f'KRW {total:,}'
    return _KRW_EXPR_PATTERN.sub(_replace, content)

def normalize_chart_table_currency_values(content: str) -> str:
    if not content or not isinstance(content, str):
        return content
    normalized = normalize_markdown_table_content(normalize_krw_expressions(content))
    lines = normalized.splitlines()
    table_lines = [line for line in lines if line.strip().startswith('|')]
    if len(table_lines) < 3:
        return normalized
    rows = []
    for line in table_lines:
        cells = split_markdown_table_row(line)
        rows.append(cells)
    header = rows[0]
    body_rows = rows[2:]
    if not header or not body_rows:
        return normalized
    default_multiplier = 10 ** 8
    column_multipliers: dict[int, int] = {}
    convert_columns: set[int] = set()
    for idx, header_cell in enumerate(header):
        if _AMOUNT_HEADER_PATTERN.search(header_cell):
            convert_columns.add(idx)
            if re.search('(trillion|조|兆)', header_cell, re.IGNORECASE):
                column_multipliers[idx] = 10 ** 12
            elif re.search('(billion|억|億)', header_cell, re.IGNORECASE):
                column_multipliers[idx] = 10 ** 8
            elif re.search('(million|백만)', header_cell, re.IGNORECASE):
                column_multipliers[idx] = 10 ** 6
            else:
                column_multipliers[idx] = default_multiplier
    if not convert_columns and header and _YEAR_HEADER_PATTERN.search(header[0]):
        numeric_columns = set(range(1, len(header)))
        for row in body_rows:
            for idx in list(numeric_columns):
                if idx >= len(row):
                    numeric_columns.discard(idx)
                    continue
                cell = row[idx]
                if cell in ('', '-'):
                    continue
                if cell.startswith('KRW '):
                    continue
                if not _PLAIN_NUMBER_PATTERN.fullmatch(cell):
                    numeric_columns.discard(idx)
        convert_columns = numeric_columns
    if not convert_columns:
        return normalized
    for row in body_rows:
        for idx in convert_columns:
            if idx >= len(row):
                continue
            cell = normalize_krw_expressions(row[idx]).strip()
            if not cell or cell == '-':
                row[idx] = cell
                continue
            if cell.startswith('KRW ') or _PERCENT_PATTERN.fullmatch(cell):
                row[idx] = cell
                continue
            if _PLAIN_NUMBER_PATTERN.fullmatch(cell):
                multiplier = column_multipliers.get(idx, default_multiplier)
                amount = int(Decimal(cell.replace(',', '')) * multiplier)
                row[idx] = f'KRW {amount:,}'
            else:
                row[idx] = cell
    rebuilt_lines = []
    for line_idx, cells in enumerate(rows):
        if line_idx == 1:
            rebuilt_lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
        else:
            rebuilt_lines.append('| ' + ' | '.join(cells) + ' |')
    rebuilt_iter = iter(rebuilt_lines)
    output_lines = []
    for line in lines:
        if line.strip().startswith('|'):
            output_lines.append(next(rebuilt_iter))
        else:
            output_lines.append(line)
    return '\n'.join(output_lines)

def merge_translated_table_with_source_values(translated_content: str, source_content: str) -> str:
    translated = normalize_markdown_table_content(translated_content)
    source = normalize_chart_table_currency_values(source_content)
    translated_lines = [line for line in translated.splitlines() if line.strip().startswith('|')]
    source_lines = [line for line in source.splitlines() if line.strip().startswith('|')]
    if len(translated_lines) < 3 or len(translated_lines) != len(source_lines):
        return translated

    def _split(line: str) -> list[str]:
        return split_markdown_table_row(line)
    translated_rows = [_split(line) for line in translated_lines]
    source_rows = [_split(line) for line in source_lines]
    if not translated_rows or len(translated_rows[0]) != len(source_rows[0]):
        return translated
    merged_rows: list[list[str]] = []
    for row_idx, (translated_row, source_row) in enumerate(zip(translated_rows, source_rows)):
        if row_idx == 1:
            merged_rows.append(['---'] * len(translated_rows[0]))
            continue
        merged_row = []
        for translated_cell, source_cell in zip(translated_row, source_row):
            if _KRW_VALUE_PATTERN.fullmatch(source_cell) or _PERCENT_PATTERN.fullmatch(source_cell) or _PLAIN_NUMBER_PATTERN.fullmatch(source_cell):
                merged_row.append(source_cell)
            else:
                merged_row.append(translated_cell)
        merged_rows.append(merged_row)
    merged_lines = ['| ' + ' | '.join(row) + ' |' for row in merged_rows]
    merged_iter = iter(merged_lines)
    output_lines = []
    for line in translated.splitlines():
        if line.strip().startswith('|'):
            output_lines.append(next(merged_iter))
        else:
            output_lines.append(line)
    return '\n'.join(output_lines)

def is_valid_markdown_table(content: str) -> bool:
    if not content or not isinstance(content, str):
        return True
    content = normalize_markdown_table_content(content)
    if not content.strip():
        return True
    if not content.strip().startswith('|'):
        return True
    lines = content.strip().split('\n')
    table_lines = [line for line in lines if line.strip().startswith('|')]
    if not table_lines:
        return True
    for line in table_lines:
        if _SEPARATOR_PATTERN.search(line):
            continue
        stripped_line = line.strip()
        if re.search('(\\|\\s+){5,}', stripped_line):
            consecutive_empty = re.findall('\\|\\s+(?=\\|)', stripped_line)
            if len(consecutive_empty) >= 5:
                logger.warning(f'Detected corrupted table with {len(consecutive_empty)} consecutive empty cells: {line[:100]}...')
                return False
    return True
import html as _html
_NATIVE_CLASS_PATTERN = re.compile('<\\|class_start\\|>\\s*(?P<cls>[^<|]*?)\\s*<\\|class_end\\|>', re.S)
_NATIVE_CONTENT_PATTERN = re.compile('<\\|content_start\\|>(?P<body>.*?)<\\|content_end\\|>', re.S)
_NATIVE_TOKEN_PATTERN = re.compile('<\\|[^|>]*\\|>')

def parse_native_chart_tokens(content):
    if not content or not isinstance(content, str):
        return None
    text = _html.unescape(content)
    if '<|content_start|>' not in text and '<|class_start|>' not in text:
        return None
    cls_match = _NATIVE_CLASS_PATTERN.search(text)
    raw_cls = cls_match.group('cls').strip().lower() if cls_match else ''
    if raw_cls.startswith('chart'):
        image_type = 'chart'
    elif raw_cls.startswith('flow'):
        image_type = 'flow'
    else:
        image_type = 'picture'
    body_match = _NATIVE_CONTENT_PATTERN.search(text)
    if body_match:
        table = body_match.group('body')
    else:
        table = text
    table = _NATIVE_TOKEN_PATTERN.sub('', table)
    table = table.replace('<p>', '').replace('</p>', '')
    table = re.sub('^!\\[', '', table.strip()).rstrip(']').strip()
    return (image_type, table)
'\nTableElementPostProcessor\n\n테이블 요소의 OTSL 태그를 HTML로 변환합니다.\n'
import html
import json
import re
from typing import Match

def looks_like_otsl(content: str) -> bool:
    return bool(content.lstrip().startswith(('<otsl>', '<fcel>', '<ecel>', '<lcel>', '<ucel>', '<xcel>', '<nl>')))

def looks_like_html_table(content: str) -> bool:
    stripped = content.strip()
    if stripped.startswith('<table'):
        return True
    lowered = stripped.lower()
    return '<table' in lowered and any((marker in lowered for marker in ('rowspan', 'colspan', '<tr', '<td')))

def normalize_span_attributes(content: str) -> str:
    return re.sub('\\b(rowspan|colspan)\\s*=\\s*(?:"|\\\')?(\\d+)(?:"|\\\')?', '\\1=\\2', content, flags=re.IGNORECASE)

def normalize_html_table_content(content: str) -> str:
    normalized = content.strip()
    for _ in range(3):
        if len(normalized) < 2 or normalized[0] != '"' or normalized[-1] != '"':
            break
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            break
        if not isinstance(parsed, str):
            break
        normalized = parsed.strip()
    normalized = html.unescape(normalized)
    normalized = normalized.replace('\\u0022', '"').replace('\\n', '')
    normalized = re.sub('\\\\+(")', '"', normalized)
    normalized = re.sub('\\\\+(/)', '\\1', normalized)
    normalized = re.sub('^\\s*```(?:html)?\\s*(.*?)\\s*```\\s*$', '\\1', normalized, flags=re.IGNORECASE | re.DOTALL)
    stripped = normalized.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        inner = stripped[1:-1].strip()
        if inner.startswith('<table'):
            normalized = inner
    return normalize_span_attributes(normalized)

def remove_dots_from_html_cells(html_content: str) -> str:
    if not html_content or not isinstance(html_content, str):
        return html_content

    def clean_dots_in_td(match: Match[str]) -> str:
        attrs = match.group(1)
        td_content = match.group(2)
        original = td_content
        td_content = re.sub('(\\.|\\u2026|\\u00B7){5,}', '', td_content)
        td_content = re.sub('(\\u00B7){5,}', '', td_content)
        if td_content != original:
            logger.debug('Removed dot patterns from TD cell')
        return f'<td{attrs}>{td_content}</td>'
    result = re.sub('<td([^>]*)>(.*?)</td>', clean_dots_in_td, html_content, flags=re.DOTALL)
    return result
'\nDataChartElementPostProcessor\n\n차트 요소를 후처리합니다 (현재는 bypass).\n'
import html
import re
import markdown
_SEPARATOR_ROW_PATTERN = re.compile('\\|\\s*:?-+:?\\s*(?:\\|\\s*:?-+:?\\s*)+\\|')

def _extract_pipe_row(segment: str, expected_pipe_count: int) -> tuple[str, str]:
    stripped = segment.lstrip()
    if not stripped.startswith('|'):
        return ('', segment)
    pipe_count = 0
    for index, char in enumerate(stripped):
        if char == '|':
            pipe_count += 1
            if pipe_count == expected_pipe_count:
                row = stripped[:index + 1].strip()
                remaining = stripped[index + 1:].lstrip()
                return (row, remaining)
    return (stripped.strip(), '')

def normalize_inline_markdown_table(content: str) -> str:
    normalized = normalize_markdown_table_content(content)
    if '\n' in normalized or not normalized.strip().startswith('|'):
        return normalized
    separator_match = _SEPARATOR_ROW_PATTERN.search(normalized)
    if not separator_match:
        return normalized
    separator_row = separator_match.group(0).strip()
    expected_pipe_count = separator_row.count('|')
    if expected_pipe_count < 2:
        return normalized
    header_segment = normalized[:separator_match.start()].strip()
    data_segment = normalized[separator_match.end():].strip()
    header_row, leftover_header = _extract_pipe_row(header_segment, expected_pipe_count)
    if not header_row or leftover_header.strip():
        return normalized
    rows = [header_row, separator_row]
    remaining = data_segment
    while remaining:
        row, remaining = _extract_pipe_row(remaining, expected_pipe_count)
        if not row:
            break
        rows.append(row)
    return '\n'.join(rows) if len(rows) >= 2 else normalized
"Pure-rule markdown postprocessing for the ``kdl_frontier`` mode.\n\nThese are GT-agnostic, structure-preserving text-surface rewrites applied to the\nfully-assembled document markdown right before the parse response is returned. They\nare the production port of the kdl-frontier-parser prototype's ``_apply_stack`` rules\n(validated on ParseBench: header-marking +0.0156 5D, title+quote +0.0132 5D).\n\nThree rules, applied in this exact order (matching the prototype):\n  1. header_mark  - mark a table's auto-detected multi-level header region as ``<th>``.\n  2. quote_fold   - fold curly quotes/apostrophes to ASCII (SemFormat literal-match fix).\n  3. title_promote- de-blockquote + promote standalone title-ish lines to ``# ``.\n\nAll three are idempotent and never reorder / merge / split elements, so document\nSTRUCTURE (and TEDS-Structure) is unchanged. Stdlib-only (no model, no I/O).\n"
import re
from collections import Counter
_TABLE_RE = re.compile('<table\\b[^>]*>.*?</table>', re.I | re.S)
_TR_RE = re.compile('(<tr\\b[^>]*>.*?</tr>)', re.I | re.S)
_COLSPAN_RE = re.compile('colspan\\s*=\\s*[\\"\']?([2-9]|\\d\\d)', re.I)

def _row_has_colspan(tr: str) -> bool:
    return bool(_COLSPAN_RE.search(tr))

def _auto_header_n(table_html: str) -> int:
    trs = _TR_RE.findall(table_html)
    if not trs or not _row_has_colspan(trs[0]):
        return 0
    n = 0
    for tr in trs:
        if _row_has_colspan(tr):
            n += 1
        else:
            break
    return min(n + 1, len(trs))

def _mark_n(table_html: str, n: int) -> str:
    out = table_html
    for tr in _TR_RE.findall(table_html)[:n]:
        m = re.sub('<td(\\s[^>]*)?>', '<th\\1>', tr, flags=re.I)
        m = re.sub('</td>', '</th>', m, flags=re.I)
        out = out.replace(tr, m, 1)
    return out

def header_mark(md: str) -> str:
    if not md:
        return md

    def _repl(match: 're.Match[str]') -> str:
        table_html = match.group(0)
        n = _auto_header_n(table_html)
        if n >= 1:
            marked = _mark_n(table_html, n)
            if marked != table_html:
                return marked
        return table_html
    return _TABLE_RE.sub(_repl, md)
_QUOTE_FOLD = {'‘': "'", '’': "'", '‚': "'", '‛': "'", '“': '"', '”': '"', '„': '"', '‟': '"'}
_QUOTE_TABLE = str.maketrans(_QUOTE_FOLD)

def quote_fold(md: str) -> str:
    if not md:
        return md
    return md.translate(_QUOTE_TABLE)
_HEADING_RE = re.compile('^\\s*#{1,6}\\s+')
_LIST_RE = re.compile('^\\s*[-*+]\\s+')
_NUMLIST_RE = re.compile('^\\s*\\d+\\.\\s+')
_TABLEROW_RE = re.compile('^\\s*\\|')
_LABELVALUE_RE = re.compile('^.{1,40}:\\s')
_BLOCKQUOTE_RE = re.compile('^\\s*>\\s+(.+)$')
_FENCE_RE = re.compile('^\\s*(```|~~~)')
_TERMINAL_PUNCT = tuple('.!?:;,')
_LETTER_RE = re.compile('[A-Za-z]')
_UPPER_RE = re.compile('[A-Z]')
_PAGE_MARKER_RE = re.compile('^\\*\\*Page\\s+\\d+\\*\\*$')
_TITLE_VARIANTS = {'aggressive': (12, 0.6, False), 'strict_caps_bq': (8, 0.85, True), 'deblockquote_only': (12, 0.6, False), 'ultra': (22, 0.25, False), 'ultra2': (30, 0.0, False)}

def _is_titleish(text: str, max_words: int, caps_ratio: float, require_all_caps: bool) -> bool:
    s = text.strip()
    if not s:
        return False
    if _HEADING_RE.match(s) or _LIST_RE.match(s) or _NUMLIST_RE.match(s) or _TABLEROW_RE.match(s):
        return False
    if len(s.split()) > max_words:
        return False
    if s.endswith(_TERMINAL_PUNCT):
        return False
    if _LABELVALUE_RE.match(s):
        return False
    letters = _LETTER_RE.findall(s)
    if not letters:
        return False
    caps_frac = len(_UPPER_RE.findall(s)) / len(letters)
    if require_all_caps:
        return caps_frac >= caps_ratio
    first_alpha = next((c for c in s if c.isalpha()), '')
    return first_alpha.isupper() or caps_frac > caps_ratio

def title_promote(md: str, variant: str='aggressive') -> str:
    if not md:
        return md
    max_words, caps_ratio, require_all_caps = _TITLE_VARIANTS[variant]
    do_promote = variant != 'deblockquote_only'
    lines = md.split('\n')
    n = len(lines)
    out = list(lines)
    in_fence = False
    for i, raw in enumerate(lines):
        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        above = lines[i - 1] if i > 0 else ''
        below = lines[i + 1] if i + 1 < n else ''
        standalone = (i == 0 or above.strip() == '') and (i + 1 >= n or below.strip() == '')
        if not standalone:
            continue
        if _PAGE_MARKER_RE.match(raw.strip()):
            continue
        bq = _BLOCKQUOTE_RE.match(raw)
        if bq:
            inner = bq.group(1)
            if _is_titleish(inner, max_words, caps_ratio, require_all_caps):
                out[i] = '# ' + inner.strip() if do_promote else inner.strip()
            continue
        if do_promote and _is_titleish(raw, max_words, caps_ratio, require_all_caps):
            out[i] = '# ' + raw.strip()
    return '\n'.join(out)
_MAX_MD_LEN = 2000000

def _looks_runaway(md: str) -> bool:
    if len(md) > _MAX_MD_LEN:
        return True
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    if len(lines) > 200:
        top = Counter(lines).most_common(1)
        if top and top[0][1] > 1000:
            return True
    return False

def postprocess_markdown(md: str, *, title_variant: str='aggressive') -> str:
    if not md or _looks_runaway(md):
        return md
    try:
        md = header_mark(md)
    except Exception:
        pass
    try:
        md = quote_fold(md)
        md = title_promote(md, variant=title_variant)
    except Exception:
        pass
    return md
_NANO_PROMPTS = {'layout': '\nLayout Detection:\n', 'text': '\nText Recognition:\n', 'table': '\nTable Recognition:\n', 'picture': '\nImage Analysis:\n', 'formula': '\nFormula Recognition:'}
_NANO_EXTRA_PAYLOAD = {'layout': {'skip_special_tokens': False, 'top_p': 0.01, 'top_k': 1, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repetition_penalty': 1.0, 'extra_args': {'no_repeat_ngram_size': 100}}, 'text': {'top_p': 0.01, 'top_k': 1, 'repetition_penalty': 1.0, 'extra_args': {'no_repeat_ngram_size': 100}}, 'table': {'skip_special_tokens': False, 'top_p': 0.01, 'top_k': 1, 'presence_penalty': 1.0, 'frequency_penalty': 0.005, 'repetition_penalty': 1.0, 'extra_args': {'no_repeat_ngram_size': 100}}, 'picture': {'top_p': 0.01, 'top_k': 1, 'repetition_penalty': 1.0, 'extra_args': {'no_repeat_ngram_size': 100}}, 'formula': None}
_NANO_MAX_TOKENS_ENV = {'layout': ('KDL_NANO_LAYOUT_MAX_TOKENS', 6000), 'text': ('KDL_NANO_TEXT_MAX_TOKENS', 2048), 'table': ('KDL_NANO_TABLE_MAX_TOKENS', 5500), 'picture': ('KDL_NANO_PICTURE_MAX_TOKENS', 4096), 'formula': ('KDL_NANO_FORMULA_MAX_TOKENS', 128)}
_NANO_LOSSLESS_STAGES = {'table'}
_TEXT_BUCKET_CATEGORIES = {'Title', 'Section-header', 'Text', 'Page-header', 'Page-footer', 'List-item', 'Caption', 'Footnote'}
_BLOCKQUOTE_CATEGORIES = {'Caption', 'Footnote', 'Page-header', 'Page-footer'}

def _nano_image_to_data_uri(image: Image.Image, *, lossless: bool=False) -> str:
    buffered = io.BytesIO()
    if lossless or image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        mime_type = 'image/png'
        image.save(buffered, format='PNG')
    else:
        if image.mode not in ('L', 'RGB', 'CMYK'):
            image = image.convert('RGB')
        mime_type = 'image/jpeg'
        image.save(buffered, format='JPEG', quality=95)
    b64 = base64.b64encode(buffered.getvalue()).decode('ascii')
    return f'data:{mime_type};base64,{b64}'

def _nano_payload(stage: str, model: str, image: Image.Image, max_output_tokens: dict[str, int] | None=None) -> dict:
    payload = {'model': model, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': _nano_image_to_data_uri(image, lossless=stage in _NANO_LOSSLESS_STAGES)}}, {'type': 'text', 'text': _NANO_PROMPTS[stage]}]}], 'temperature': 0.0, 'max_tokens': int((max_output_tokens or {}).get(stage, os.getenv(_NANO_MAX_TOKENS_ENV[stage][0], str(_NANO_MAX_TOKENS_ENV[stage][1]))))}
    extra = _NANO_EXTRA_PAYLOAD[stage]
    if extra is not None:
        payload.update(extra)
    return payload

async def _nano_chat(client: httpx.AsyncClient, url: str, payload: dict, semaphore: asyncio.Semaphore, max_retries: int=2, *, stage: str | None=None, usage: NanoUsage | None=None, sequence_limiter: SequenceLimiter | None=None, headers: dict[str, str] | None=None) -> str | None:
    last_exc: Exception | None = None
    result: str | None = None
    retries = 0
    started = time.perf_counter()
    limiter = sequence_limiter or SequenceLimiter(1)
    async with semaphore, limiter.hold(1):
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(url, json={**payload, 'chat_template_kwargs': {'enable_thinking': False}}, headers=headers)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f'{resp.status_code}', request=resp.request, response=resp)
                if resp.status_code >= 400:
                    logger.warning('4xx from endpoint (not retried): %s', resp.text[:200])
                    break
                data = resp.json()
                result = data['choices'][0]['message']['content']
                break
            except (httpx.HTTPError, asyncio.TimeoutError, KeyError, ValueError) as e:
                last_exc = e
                if attempt < max_retries:
                    retries += 1
                    await asyncio.sleep(min(10.0, 2.0 * 2 ** attempt))
    if result is None and last_exc is not None:
        logger.warning('stage request failed after retries: %s', last_exc)
    if usage is not None and stage is not None:
        usage.record(stage, sequences=1, latency_ms=(time.perf_counter() - started) * 1000.0, failures=int(result is None), retries=retries)
    return result

async def _nano_chat_batch(client: httpx.AsyncClient, url: str, payloads: List[dict], semaphore: asyncio.Semaphore, max_retries: int=2, *, stage: str | None=None, usage: NanoUsage | None=None, sequence_limiter: SequenceLimiter | None=None, headers: dict[str, str] | None=None) -> List[str | None]:
    if not payloads:
        return []
    batch_payload = {**payloads[0], 'messages': [payload['messages'] for payload in payloads], 'chat_template_kwargs': {'enable_thinking': False}}
    results: List[str | None] = [None] * len(payloads)
    retries = 0
    last_exc: Exception | None = None
    started = time.perf_counter()
    limiter = sequence_limiter or SequenceLimiter(len(payloads))
    async with semaphore, limiter.hold(len(payloads)):
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(url, json=batch_payload, headers=headers)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f'{resp.status_code}', request=resp.request, response=resp)
                if resp.status_code >= 400:
                    logger.warning('4xx from batch endpoint (not retried): %s', resp.text[:200])
                    break
                choices = resp.json().get('choices') or []
                for fallback_index, choice in enumerate(choices):
                    index = int(choice.get('index', fallback_index))
                    if 0 <= index < len(results):
                        results[index] = (choice.get('message') or {}).get('content')
                break
            except (httpx.HTTPError, asyncio.TimeoutError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    retries += 1
                    await asyncio.sleep(min(10.0, 2.0 * 2 ** attempt))
    if all((result is None for result in results)) and last_exc is not None:
        logger.warning('batch request failed after retries: %s', last_exc)
    if usage is not None and stage is not None:
        usage.record(stage, sequences=len(payloads), latency_ms=(time.perf_counter() - started) * 1000.0, failures=sum((result is None for result in results)), retries=retries)
    return results

def _nano_group_by_bucket(content: List[Dict[str, Any]], original_image: Image.Image) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {'text': [], 'table': [], 'picture': [], 'formula': []}
    im_w, im_h = original_image.size
    for item in content:
        cat = item.get('category', 'Text')
        mapped_cat = layout_recognition_bucket(cat)
        bbox = item.get('bbox', [])
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = bbox
            left = int(round(x1 * im_w))
            upper = int(round(y1 * im_h))
            right = int(round(x2 * im_w))
            lower = int(round(y2 * im_h))
            left = max(0, min(left, im_w))
            upper = max(0, min(upper, im_h))
            right = max(left + 1, min(right, im_w))
            lower = max(upper + 1, min(lower, im_h))
            pixel_width = right - left
            pixel_height = lower - upper
            if pixel_width < 5 or pixel_height < 5:
                continue
            cropped_img = original_image.crop((left, upper, right, lower))
            preprocessed_img = preprocess_for_vlm(cropped_img)
            if is_monochromatic(preprocessed_img):
                continue
        except Exception:
            continue
        element_info = {'bbox': bbox, 'category': cat, 'layout_order': item.get('layout_order', 0), 'page_number': item.get('page_number', 1), 'preprocessed_image': preprocessed_img}
        if 'job_id' in item:
            element_info['job_id'] = item['job_id']
        if 'angle' in item:
            element_info['angle'] = item.get('angle')
        if mapped_cat == 'picture':
            element_info['cropped_image'] = cropped_img
        result[mapped_cat].append(element_info)
    return result

def _nano_group_layout_items(content: List[Dict[str, Any]], page_number: int) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {'text': [], 'table': [], 'picture': [], 'formula': []}
    for item in content:
        bbox = item.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            numeric_bbox = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = numeric_bbox
        if not (x1 < x2 and y1 < y2):
            continue
        category = str(item.get('category') or 'Text')
        element: Dict[str, Any] = {'bbox': numeric_bbox, 'category': category, 'layout_order': int(item.get('layout_order', 0)), 'page_number': page_number}
        if 'angle' in item:
            element['angle'] = item.get('angle')
        result[layout_recognition_bucket(category)].append(element)
    return result

def _nano_is_single_clean_otsl(content: Any) -> bool:
    return isinstance(content, str) and content.count('<otsl>') == 1 and ('<fcel>' in content or '<ecel>' in content)

def _nano_apply_picture_result(el: Dict[str, Any], content: str | None) -> None:
    if content is None:
        el['content'] = ''
        return
    content_str = content.strip()
    native_parsed = parse_native_chart_tokens(content_str)
    if native_parsed is not None:
        image_type, table = native_parsed
        if table:
            table = normalize_krw_expressions(table)
        if image_type == 'chart':
            el['category'] = 'Chart'
        elif image_type == 'flow':
            el['category'] = 'Flowchart'
        normalized = normalize_markdown_table_content(table)
        if is_valid_markdown_table(normalized) and normalized.strip().startswith('|'):
            el['content'] = normalized
        else:
            el['content'] = table
        return
    if content_str.startswith('|') and is_valid_markdown_table(content_str):
        el['content'] = normalize_markdown_table_content(content_str)
        el['category'] = 'Chart'
    else:
        el['content'] = content_str

def _nano_postprocess_element(el: Dict[str, Any]) -> None:
    content = el.get('content') or ''
    category = el.get('category', 'Text')
    if not content.strip():
        return
    try:
        if category == 'Table':
            if looks_like_html_table(content):
                el['content'] = normalize_html_table_content(content)
            elif looks_like_otsl(content):
                cleaned = truncate_repetitive_content(content, preserve_line_breaks=True)
                html_content = convert_otsl_to_html_v2(cleaned)
                html_content = normalize_span_attributes(html_content)
                el['content'] = remove_dots_from_html_cells(html_content)
        elif category in _TEXT_BUCKET_CATEGORIES:
            el['content'] = truncate_repetitive_content(content)
        elif category in ('Picture', 'Flowchart'):
            el['content'] = f'![{truncate_repetitive_content(content)}]'
        elif category == 'Chart':
            cleaned = normalize_inline_markdown_table(content)
            el['content'] = _markdown_lib.markdown(html.escape(cleaned), extensions=['tables'])
        elif category == 'Formula':
            cleaned = truncate_repetitive_content(content)
            if '\\(' in cleaned and '\\)' in cleaned or ('\\[' in cleaned and '\\]' in cleaned):
                cleaned = cleaned.replace('$', '')
                cleaned = cleaned.replace('\\(', ' $ ').replace('\\)', ' $ ').replace('\\[', ' $$ ').replace('\\]', ' $$ ')
            el['content'] = cleaned
    except Exception as e:
        logger.warning('element postprocess failed (%s): %s', category, e)
_HTML_STRIKE_OPEN_RE = re.compile('<\\s*(?:s|strike|del)(?:\\s[^>]*)?>', re.IGNORECASE)
_HTML_STRIKE_CLOSE_RE = re.compile('<\\s*/\\s*(?:s|strike|del)\\s*>', re.IGNORECASE)
_LEADING_MD_HEADING_RE = re.compile('^[ \\t]*#{1,6}[ \\t]+')

def _strip_leading_heading_marker(content: str) -> str:
    if not content:
        return content
    return _LEADING_MD_HEADING_RE.sub('', content, count=1)

def _preserve_inline_markup(content: str) -> str:
    if not content:
        return content
    content = _HTML_STRIKE_OPEN_RE.sub('~~', content)
    content = _HTML_STRIKE_CLOSE_RE.sub('~~', content)
    return content

def _nano_format_blockquote(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ''
    lines = stripped.splitlines()
    quoted = [f'> {lines[0]}']
    quoted.extend((f'> {line}' if line else '>' for line in lines[1:]))
    return '\n'.join(quoted)

def _nano_format_formula(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ''
    if stripped.startswith('```'):
        return stripped
    return f'```latex\n{stripped}\n```'

def _nano_image_markdown(el: Dict[str, Any]) -> str:
    image_src = el.get('picture_path') or ''
    if not image_src:
        return ''
    alt = {'Chart': 'chart', 'Flowchart': 'flowchart'}.get(el['category'], 'picture')
    return f'![{alt}]({image_src})'

def _nano_format_element(el: Dict[str, Any]) -> str:
    category = el.get('category', 'Text')
    content = el.get('content') or ''
    if category in ('Title', 'Section-header'):
        prefix = '#' if category == 'Title' else '##'
        return f'{prefix} {_preserve_inline_markup(_strip_leading_heading_marker(content))}'
    if category == 'Table':
        return content
    if category == 'Chart':
        image_markdown = _nano_image_markdown(el)
        blocks = [b for b in (image_markdown, content) if b]
        return '\n\n'.join(blocks)
    if category in ('Picture', 'Flowchart'):
        image_markdown = _nano_image_markdown(el)
        if not image_markdown:
            return content
        stripped = content.strip()
        if stripped.startswith('|') or stripped.startswith('<table'):
            return f'{image_markdown}\n\n{stripped}'
        if content:
            return f'{image_markdown}\n\n{content}'
        return image_markdown
    if category in _BLOCKQUOTE_CATEGORIES:
        return _nano_format_blockquote(content)
    if category == 'Formula':
        return _nano_format_formula(content)
    if category == 'List-item':
        c = _preserve_inline_markup(_strip_leading_heading_marker(content.strip()))
        return f'- {c}' if c else ''
    return _preserve_inline_markup(_strip_leading_heading_marker(content.strip()))

def _nano_assemble_markdown(elements: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    valid = [e for e in sorted(elements, key=lambda e: (e.get('page_number', 1), e.get('layout_order', 0))) if isinstance(e.get('page_number'), int) and (not isinstance(e.get('page_number'), bool)) and (e.get('page_number', 0) >= 1)]
    blocks: List[Tuple[int, str]] = []
    index = 0
    while index < len(valid):
        el = valid[index]
        page = el['page_number']
        if el.get('category') == 'List-item':
            items = []
            while index < len(valid) and valid[index].get('category') == 'List-item' and (valid[index]['page_number'] == page):
                formatted = _nano_format_element(valid[index])
                if formatted:
                    items.append(formatted)
                index += 1
            content = '\n'.join(items).strip()
        else:
            content = _nano_format_element(el).strip()
            index += 1
        if content:
            blocks.append((page, content))
    md_parts: List[str] = []
    current_page: int | None = None
    for page, content in blocks:
        if current_page is not None and page != current_page:
            md_parts.append(f'---\n\n**Page {page}**')
        md_parts.append(content)
        current_page = page
    pages_md: Dict[int, List[str]] = {}
    for page, content in blocks:
        pages_md.setdefault(page, []).append(content)
    markdown_pages = [{'page_number': page, 'content': '\n\n'.join(parts)} for page, parts in sorted(pages_md.items())]
    return ('\n\n'.join(md_parts), markdown_pages)

class _NanoEngine:

    def __init__(self, endpoint_url: str, model: str, bbox_max_concurrent: int, timeout_s: float, *, page_max_concurrent: int=1, max_retries: int=2, max_output_tokens: dict[str, int] | None=None, text_router: Any | None=None, request_batch_size: int=1, max_model_sequences: int=32):
        base = endpoint_url.rstrip('/')
        self._url = base + '/chat/completions' if base.endswith('/v1') else base + '/v1/chat/completions'
        self._model = model
        self._bbox_max_concurrent = bbox_max_concurrent
        self._page_max_concurrent = page_max_concurrent
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._request_batch_size = max(1, int(request_batch_size))
        self._max_model_sequences = max(1, int(max_model_sequences))
        if self._request_batch_size > self._max_model_sequences:
            raise ValueError('request_batch_size must not exceed max_model_sequences')
        self._batch_url = self._url + '/batch'
        self._headers = {'ngrok-skip-browser-warning': 'true'}
        api_key = os.getenv('KDL_NANO_API_KEY') or os.getenv('VLLM_API_KEY')
        if api_key:
            self._headers['Authorization'] = f'Bearer {api_key}'
        self._text_router = text_router

    async def layout_batch(self, client: httpx.AsyncClient, pages: List[tuple[Image.Image, int]], request_semaphore: asyncio.Semaphore, *, sequence_limiter: SequenceLimiter, usage: NanoUsage) -> List[Dict[str, List[Dict[str, Any]]]]:
        grouped: List[Dict[str, List[Dict[str, Any]]]] = [{'text': [], 'table': [], 'picture': [], 'formula': []} for _ in pages]
        eligible: List[tuple[int, Image.Image, int]] = []
        for index, (image, page_number) in enumerate(pages):
            if min(image.size) < 32:
                continue
            try:
                if analyze_page_content(image).is_blank:
                    continue
            except Exception:
                pass
            eligible.append((index, image, page_number))
        if not eligible:
            return grouped
        payloads = [_nano_payload('layout', self._model, prepare_native_layout_image(image), self._max_output_tokens) for _, image, _ in eligible]
        if len(payloads) == 1 and self._request_batch_size <= 1:
            contents = [await _nano_chat(client, self._url, payloads[0], request_semaphore, self._max_retries, stage='layout', usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)]
        else:
            contents = await _nano_chat_batch(client, self._batch_url, payloads, request_semaphore, self._max_retries, stage='layout', usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)
        missing_indexes = [index for index, content in enumerate(contents) if content is None]
        if missing_indexes:
            usage.batch_fallback_sequences['layout'] += len(missing_indexes)
            fallback_contents = await asyncio.gather(*(_nano_chat(client, self._url, payloads[index], request_semaphore, self._max_retries, stage='layout', usage=usage, sequence_limiter=sequence_limiter, headers=self._headers) for index in missing_indexes))
            for index, content in zip(missing_indexes, fallback_contents, strict=True):
                contents[index] = content
            usage.batch_fallback_recovered_sequences['layout'] += sum((content is not None for content in fallback_contents))
        for (output_index, _, page_number), content in zip(eligible, contents, strict=True):
            if not content or not content.strip():
                continue
            if not is_native_layout_response(content):
                logger.warning('page %d: layout response has no <|box_start|> tokens', page_number)
                continue
            grouped[output_index] = _nano_group_layout_items(parse_native_layout_tokens(content), page_number)
        return grouped

    async def recognize_prepared_batch(self, client: httpx.AsyncClient, stage: str, elements: List[Dict[str, Any]], request_semaphore: asyncio.Semaphore, *, sequence_limiter: SequenceLimiter, usage: NanoUsage, image_key: str='preprocessed_image', fullpage_table: bool=False) -> List[Dict[str, Any]]:
        if not elements:
            return []
        payloads = [_nano_payload(stage, self._model, element[image_key], self._max_output_tokens) for element in elements]
        if len(payloads) == 1 and self._request_batch_size <= 1:
            contents = [await _nano_chat(client, self._url, payloads[0], request_semaphore, self._max_retries, stage=stage, usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)]
        else:
            contents = await _nano_chat_batch(client, self._batch_url, payloads, request_semaphore, self._max_retries, stage=stage, usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)
        missing_indexes = [index for index, content in enumerate(contents) if content is None]
        if missing_indexes:
            usage.batch_fallback_sequences[stage] += len(missing_indexes)
            fallback_contents = await asyncio.gather(*(_nano_chat(client, self._url, payloads[index], request_semaphore, self._max_retries, stage=stage, usage=usage, sequence_limiter=sequence_limiter, headers=self._headers) for index in missing_indexes))
            for index, content in zip(missing_indexes, fallback_contents, strict=True):
                contents[index] = content
            usage.batch_fallback_recovered_sequences[stage] += sum((content is not None for content in fallback_contents))
        table_fallbacks: List[Dict[str, Any]] = []
        for element, content in zip(elements, contents, strict=True):
            if fullpage_table:
                if content is not None and _nano_is_single_clean_otsl(content):
                    element['content'] = content
                    element['recognition_source'] = 'kdl_table'
                else:
                    table_fallbacks.append(element)
                continue
            if stage == 'picture':
                _nano_apply_picture_result(element, content)
            else:
                element['content'] = content if content is not None else ''
            element['recognition_source'] = f'kdl_{stage}'
            if content is None:
                usage.unrecovered_sequences[stage] += 1
        return table_fallbacks

    async def parse_pages(self, page_images: List[Image.Image], *, routing_context: Any | None=None) -> dict:
        page_semaphore = asyncio.Semaphore(self._page_max_concurrent)
        layout_semaphore = asyncio.Semaphore(self._page_max_concurrent)
        bbox_semaphore = asyncio.Semaphore(self._bbox_max_concurrent)
        sequence_limiter = SequenceLimiter(self._max_model_sequences)
        usage = NanoUsage()

        async def parse_page(client: httpx.AsyncClient, image: Image.Image, page_no: int) -> List[Dict[str, Any]]:
            async with page_semaphore:
                return await self._parse_page(client, layout_semaphore, bbox_semaphore, normalize_image_mode(image, 'RGB'), page_no, routing_context=routing_context, sequence_limiter=sequence_limiter, usage=usage)
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            pages = await asyncio.gather(*(parse_page(client, image, page_no) for page_no, image in enumerate(page_images, start=1)))
        elements = [element for page in pages for element in page]
        raw = self.finalize_elements(elements)
        raw['usage'] = usage.snapshot()
        return raw

    def finalize_elements(self, elements: List[Dict[str, Any]]) -> dict:
        for el in elements:
            _nano_postprocess_element(el)
        full_md, markdown_pages = _nano_assemble_markdown(elements)
        full_md = postprocess_markdown(full_md)
        for page in markdown_pages:
            page['content'] = postprocess_markdown(page['content'])
        pages_payload: Dict[int, List[Dict[str, Any]]] = {}
        for el in elements:
            payload = {'category': el.get('category', 'Text'), 'bbox': el.get('bbox'), 'content': el.get('content') or '', 'layout_order': el.get('layout_order', 0)}
            if el.get('recognition_source'):
                payload['recognition_source'] = el['recognition_source']
            pages_payload.setdefault(el['page_number'], []).append(payload)
        return {'markdown': full_md, 'markdown_pages': markdown_pages, 'pages': [{'page_number': n, 'elements': els} for n, els in sorted(pages_payload.items())]}

    async def _parse_page(self, client: httpx.AsyncClient, layout_semaphore: asyncio.Semaphore, bbox_semaphore: asyncio.Semaphore, image: Image.Image, page_no: int, *, routing_context: Any | None=None, sequence_limiter: SequenceLimiter | None=None, usage: NanoUsage | None=None) -> List[Dict[str, Any]]:
        w, h = image.size
        if min(w, h) < 32:
            return []
        try:
            if analyze_page_content(image).is_blank:
                return []
        except Exception:
            pass
        layout_image = prepare_native_layout_image(image)
        layout_content = await _nano_chat(client, self._url, _nano_payload('layout', self._model, layout_image, self._max_output_tokens), layout_semaphore, self._max_retries, stage='layout', usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)
        if not layout_content or not layout_content.strip():
            return []
        if not is_native_layout_response(layout_content):
            logger.warning('page %d: layout response has no <|box_start|> tokens', page_no)
            return []
        items = parse_native_layout_tokens(layout_content)
        for item in items:
            item['page_number'] = page_no
        buckets = _nano_group_by_bucket(items, image)
        fullpage_table = preprocess_for_vlm(image) if len(buckets['table']) == 1 else None
        routed_text_indices: set[int] = set()
        if self._text_router is not None and buckets['text']:
            try:
                routed = await self._text_router.route_text_regions(routing_context, page_no, buckets['text'])
                routed_text_indices = {int(index) for index in routed or set()}
            except Exception as exc:
                logger.warning('page %d: native text routing failed; using KDL: %s', page_no, exc)
        tasks = []

        async def recognize(stage: str, el: Dict[str, Any]) -> None:
            pre = el.get('preprocessed_image')
            if pre is None:
                el['content'] = ''
                if self._text_router is not None:
                    el['recognition_source'] = f'kdl_{stage}'
                return
            if stage == 'picture' and (pre.width < 25 or pre.height < 25):
                el['content'] = ''
                if self._text_router is not None:
                    el['recognition_source'] = f'kdl_{stage}'
                return
            content = await _nano_chat(client, self._url, _nano_payload(stage, self._model, pre, self._max_output_tokens), bbox_semaphore, self._max_retries, stage=stage, usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)
            if stage == 'picture':
                _nano_apply_picture_result(el, content)
            else:
                el['content'] = content if content is not None else ''
            if content is None and usage is not None:
                usage.unrecovered_sequences[stage] += 1
            if self._text_router is not None:
                el['recognition_source'] = f'kdl_{stage}'

        async def recognize_batch(stage: str, elements: List[Dict[str, Any]]) -> None:
            eligible: List[Dict[str, Any]] = []
            for el in elements:
                pre = el.get('preprocessed_image')
                if pre is None:
                    el['content'] = ''
                    if self._text_router is not None:
                        el['recognition_source'] = f'kdl_{stage}'
                    continue
                if stage == 'picture' and (pre.width < 25 or pre.height < 25):
                    el['content'] = ''
                    if self._text_router is not None:
                        el['recognition_source'] = f'kdl_{stage}'
                    continue
                eligible.append(el)
            if not eligible:
                return
            contents = await _nano_chat_batch(client, self._batch_url, [_nano_payload(stage, self._model, el['preprocessed_image'], self._max_output_tokens) for el in eligible], bbox_semaphore, self._max_retries, stage=stage, usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)
            missing_indexes = [index for index, content in enumerate(contents) if content is None]
            if missing_indexes:
                if usage is not None:
                    usage.batch_fallback_sequences[stage] += len(missing_indexes)
                fallback_contents = await asyncio.gather(*(_nano_chat(client, self._url, _nano_payload(stage, self._model, eligible[index]['preprocessed_image'], self._max_output_tokens), bbox_semaphore, self._max_retries, stage=stage, usage=usage, sequence_limiter=sequence_limiter, headers=self._headers) for index in missing_indexes))
                for index, content in zip(missing_indexes, fallback_contents, strict=True):
                    contents[index] = content
                if usage is not None:
                    usage.batch_fallback_recovered_sequences[stage] += sum((content is not None for content in fallback_contents))
            for el, content in zip(eligible, contents, strict=True):
                if stage == 'picture':
                    _nano_apply_picture_result(el, content)
                else:
                    el['content'] = content if content is not None else ''
                if content is None and usage is not None:
                    usage.unrecovered_sequences[stage] += 1
                if self._text_router is not None:
                    el['recognition_source'] = f'kdl_{stage}'

        async def recognize_table_fullpage(el: Dict[str, Any]) -> None:
            content = await _nano_chat(client, self._url, _nano_payload('table', self._model, fullpage_table, self._max_output_tokens), bbox_semaphore, self._max_retries, stage='table', usage=usage, sequence_limiter=sequence_limiter, headers=self._headers)
            if content is not None and _nano_is_single_clean_otsl(content):
                el['content'] = content
                if self._text_router is not None:
                    el['recognition_source'] = 'kdl_table'
                return
            await recognize('table', el)

        def add_recognition_tasks(stage: str, elements: List[Dict[str, Any]]) -> None:
            if self._request_batch_size <= 1:
                tasks.extend((recognize(stage, el) for el in elements))
                return
            for start in range(0, len(elements), self._request_batch_size):
                tasks.append(recognize_batch(stage, elements[start:start + self._request_batch_size]))
        text_elements = [el for index, el in enumerate(buckets['text']) if index not in routed_text_indices]
        add_recognition_tasks('text', text_elements)
        remaining_tables: List[Dict[str, Any]] = []
        for i, el in enumerate(buckets['table']):
            if fullpage_table is not None and i == 0:
                tasks.append(recognize_table_fullpage(el))
            else:
                remaining_tables.append(el)
        add_recognition_tasks('table', remaining_tables)
        add_recognition_tasks('picture', buckets['picture'])
        add_recognition_tasks('formula', buckets['formula'])
        await asyncio.gather(*tasks)
        page_elements: List[Dict[str, Any]] = []
        picture_idx = 0
        for bucket_name in ('text', 'table', 'picture', 'formula'):
            for el in buckets[bucket_name]:
                el.pop('preprocessed_image', None)
                cropped = el.pop('cropped_image', None)
                if bucket_name == 'picture':
                    if cropped is not None and cropped.width >= 25 and (cropped.height >= 25):
                        el['picture_path'] = f'artifacts/cropped_pictures/page_{page_no:03d}_picture_{picture_idx:03d}.png'
                    picture_idx += 1
                page_elements.append(el)
        return page_elements
NanoEngine = _NanoEngine
__all__ = ['NanoEngine', 'NanoUsage', 'SequenceLimiter']
