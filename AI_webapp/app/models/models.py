from pydantic import BaseModel
from typing import List, Optional


class ModeLInput(BaseModel):
    year: List[int]
    interest_rate: List[float]


class ModeLOutput(BaseModel):
    result: List[float]
