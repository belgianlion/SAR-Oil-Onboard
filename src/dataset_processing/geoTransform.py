from dataclasses import dataclass
from lxml import etree


@dataclass
class GeoTransform():
    x_origin: float
    pixel_width: float
    row_rotation: float
    y_origin: float
    column_rotation: float
    pixel_height: float

    def fromXml(self, xml_data: str):
        root = etree.fromstring(xml_data)
        geoTransform_entry = root.find("GeoTransform")
        geoTransform_elements = geoTransform_entry.text.strip().split(", ")
        self.x_origin = float(geoTransform_elements[0])
        self.pixel_width = float(geoTransform_elements[1])
        self.row_rotation = float(geoTransform_elements[2])
        self.y_origin = float(geoTransform_elements[3])
        self.column_rotation = float(geoTransform_elements[4])
        self.pixel_height = float(geoTransform_elements[5])