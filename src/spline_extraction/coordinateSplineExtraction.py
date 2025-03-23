from typing import List, Tuple

from src.dataset_processing.geoTransform import GeoTransform

class BSplineExtraction:
    def __init__(self, aux_xml_file: str, num_points: int = 20):
        self.num_points = num_points
        # self.geo_transform = GeoTransform().fromXml(aux_xml_file)
        self.b_spline_extractor = BSplineExtraction(num_points)

    def extract_coordinates(self, image) -> List[Tuple[float, float]]:
        splines = self.b_spline_extractor.extract_spline(image)
        