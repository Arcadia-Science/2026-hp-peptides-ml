from typing import Optional

from pydantic import BaseModel, Field, model_validator


class GeometryRequest(BaseModel):
    dataset: Optional[str] = None
    molecule_id: Optional[int] = None
    pos: Optional[list[list[float]]] = None
    z: Optional[list[int]] = None

    @model_validator(mode="after")
    def validate_input(self):
        has_inline = self.pos is not None and self.z is not None
        has_db_ref = self.dataset is not None and self.molecule_id is not None
        if not has_inline and not has_db_ref:
            raise ValueError("Provide either pos+z or dataset+molecule_id")
        return self


class NmrAggregateRequest(BaseModel):
    sc: list[float]
    sh: list[float]
    indexc: list[int] = Field(..., description="Indices mapping carbons to environments")
    indexh: list[int] = Field(..., description="Indices mapping hydrogens to environments")
