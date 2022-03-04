from dataclasses import dataclass
from typing import Union, List


@dataclass
class BaseInfo:
    name: Union[str, None]
    id: int


@dataclass
class DetectionInfo(BaseInfo):
    x: float
    y: float
    w: float
    h: float


@dataclass
class ImageInfo(BaseInfo):
    filename: Union[str, None]
    path: Union[str, None]
    width: Union[int, None]
    height: Union[int, None]
    detections: List[DetectionInfo]
    split: Union[str, None]


@dataclass
class ImagesInfo(BaseInfo):
    images: List[ImageInfo]

    def __add__(self, other):
        return ImagesInfo(images=self.images + other.images, id=self.id, name=self.name)





