from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Spline():
    points: List[Tuple[int, int]]
    tck: List
    linspace: List

