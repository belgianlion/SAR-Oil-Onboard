import xml.etree.ElementTree as etree

class ImageXmlData():
    def __init__(self, xml_jpg_image_path):
        parsed_xml_tree = etree.parse(xml_jpg_image_path)
        root = parsed_xml_tree.getroot()
        for child in root:
            if child.tag == "GeoTransform":
                transform_values = [float(x) for x in child.text.split(',')]
                self.x_origin = transform_values[0]
                self.pixel_width = transform_values[1]
                self.x_rotation = transform_values[2]
                self.y_origin = transform_values[3]
                self.y_rotation = transform_values[4]
                self.pixel_height = transform_values[5]

