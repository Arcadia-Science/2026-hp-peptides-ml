from typing import List

from pydantic import BaseModel


class GeometryInput(BaseModel):
    pos: List[List[float]]
    z: List[int]


class NmrAggregateInput(BaseModel):
    sc: List[float]
    sh: List[float]
    indexc: List[int]
    indexh: List[int]
