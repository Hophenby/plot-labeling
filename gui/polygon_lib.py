
from PySide6.QtCore import Qt, QPoint, QRect, QRectF
from PySide6.QtGui import (
    QPixmap, 
    QPainter,
    QPolygonF,
    QPainterPath,
    )
import numpy as np
import cv2

def qpixmap_to_cvimage(qpixmap:QPixmap) -> np.ndarray:
    if qpixmap.isNull():
        return None
    image=qpixmap.copy().toImage()
    width,heith=image.width(),image.height()
    buffer=image.bits()
    img=np.frombuffer(buffer,dtype=np.uint8).reshape((heith,width,4))
    img=cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    return img

class SelectedArea:
    polygons: list[QPolygonF]
    label: int = 0
    def __init__(self, polygons:list[QPolygonF]=None, rectangles:list[QRectF|QRect]=None, label=0):
        self.label = label
        if polygons is None:
            polygons = []
        self.polygons = polygons
        if rectangles is not None:
            self.polygons.extend([QPolygonF(rect) for rect in rectangles])

    def __iter__(self):
        return iter(self.polygons)
    
    def __contains__(self,point:QPoint):
        return self.containsPoint(point)
        
    def add_polygon(self,polygon:QPolygonF):
        self.polygons.append(polygon)

    def add_rect(self,rect:QRect|QRectF):
        self.polygons.append(QPolygonF(rect))

    def merge(self,other: 'SelectedArea'):
        if other.label != self.label:
            raise Warning("Merging different labels")
        self.polygons.extend(other.polygons)

    def remove_polygon(self,polygon:QPolygonF):
        self.polygons.remove(polygon)

    def area(self):
        def polygon_area(polygon:QPolygonF):
            area = 0
            for i in range(len(polygon)):
                x1, y1 = polygon[i].x(), polygon[i].y()
                x2, y2 = polygon[(i + 1) % len(polygon)].x(), polygon[(i + 1) % len(polygon)].y()
                area += (x1 * y2 - x2 * y1)
            return abs(area) / 2
        area = 0
        for polygon in self.polygons:
            area += polygon_area(polygon)
        return area
    
    def containsPoint(self,point:QPoint):
        for polygon in self.polygons:
            if polygon.containsPoint(point, Qt.FillRule.WindingFill):
                return True
        return False
    
    def draw(self,painter:QPainter):
        for polygon in self.polygons:
            painter.drawPolygon(polygon)

    def addToFillPath(self,fill_path:QPainterPath):
        for polygon in self.polygons:
            fill_path.addPolygon(polygon)

    def translated(self,dx:int,dy:int):
        return SelectedArea([polygon.translated(dx,dy) for polygon in self.polygons],label=self.label)
    
    def to_yolo(self,image_width:int,image_height:int):
        """Convert to YOLO format
        The dataset label format used for training YOLO segmentation models is as follows:

        One text file per image: Each image in the dataset has a corresponding text file with the same
        name as the image file and the ".txt" extension.
        One row per object: Each row in the text file corresponds to one object instance in the image.
        Object information per row: Each row contains the following information about the object instance:
        Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
        Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.
        The format for a single row in the segmentation dataset file is as follows:
        <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
        In this format, <class-index> is the index of the class for the object, and <x1> <y1> <x2> <y2> ... <xn> <yn> are 
        the bounding coordinates of the object's segmentation mask. The coordinates are separated by spaces."""
        yolos = []
        for polygon in self.polygons:
            yolo = [str(self.label)]
            for point in polygon:
                x = point.x() / image_width
                y = point.y() / image_height
                yolo.append(f"{x:.5f}")
                yolo.append(f"{y:.5f}")
            yolos.append(" ".join(yolo))
        return yolos
    
    def to_yolo_txt(self,image_width:int,image_height:int,filename:str):
        with open(filename,"w") as f:
            yolos = self.to_yolo(image_width,image_height)
            f.write("\n".join(yolos))
