import asyncio
import os
import time
import typing
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton, 
    QVBoxLayout,
    QWidget,
    QLabel,
    QScrollArea,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsOpacityEffect,
    )
from PySide6.QtCore import Slot, Qt, QSize, QPoint, QEvent, QByteArray, QRect, QTimer, QThread, QRunnable, QThreadPool, QRectF
from PySide6.QtGui import (
    QKeyEvent,
    QPixmap, 
    QResizeEvent,
    QPainter,
    QColor,
    QPolygonF,
    QMouseEvent,
    QPainterPath,
    QImage,
    QRegion,
    )

import cv2,numpy as np
import sys
sys.path.append("..")
import torch

from typing import overload
from seg_lib import label_names, seg_cv2_yolo
from polygon_lib import SelectedArea, qpixmap_to_cvimage

white_thres = 230


class MaskArea(QScrollArea):
    @property
    def current_label(self):
        return self.layer1.current_label
    
    def __init__(self,pic_path:str=""):
        super().__init__()
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.layer1pixmap = QPixmap(pic_path)
        self.layer1 = MaskAreaContent(pic_path)
        self.layer1.setPixmap(self.layer1pixmap)

        self.setWidget(self.layer1)
        self.setMouseTracking(True)
        #self.polygon = QPolygonF()

        self.keyPressEvent = self.layer1.keyPressEvent
        self.keyReleaseEvent = self.layer1.keyReleaseEvent
        self.save_processed_image = self.layer1.save_processed_image
        self.save_yolo_seg = self.layer1.save_yolo_seg
        self.change_pic = self.layer1.change_pic

        self.prev_label = self.layer1.prev_label
        self.next_label = self.layer1.next_label

    def resizeEvent(self, arg__1: QResizeEvent) -> None:
        size_width =arg__1.size().width()
        self.layer1.setPixmap(
            self.layer1.pixmap().scaled(
                QSize(size_width, self.layer1pixmap.height()+3000),
                  Qt.AspectRatioMode.KeepAspectRatio, 
                  Qt.TransformationMode.FastTransformation))
        return super().resizeEvent(arg__1)

    async def boot(self):
        pass

class MaskAreaContent(QLabel):
    label_area_map: dict[int, 'SelectedArea']
    """The content of the MaskArea"""
    def __init__(self,pic_path:str=None):
        super().__init__()
        self.original_pixmap = QPixmap(pic_path)
        self.processed_pixmap = QPixmap(pic_path)
        self.show_pixmap = QPixmap(pic_path) 
        self.setPixmap(self.show_pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setMouseTracking(True)

        self.maskMode = "polygon" # "polygon", "rectangle", "auto"

        self.globalGeometry = self.geometry()
        self.pixmapFrame = QRect(0, 0, self.original_pixmap.width(), self.original_pixmap.height())

        self.label_area_map = {}
        self.current_label = 0

        def get_or_create_label_area(label:int):
            if label not in self.label_area_map or self.label_area_map[label] is None:
                self.label_area_map[label] = SelectedArea(label= label)
            return self.label_area_map[label]
        
        self.selected_area = lambda: get_or_create_label_area(self.current_label)

        self.polygon_cache = QPolygonF()
        self.rectangle_cache = QRect()

        self.pixmapscale_factor=self.pixmap().width()/(self.original_pixmap.width()+0.01)

        def change_label(d_label:int):
            self.current_label = (self.current_label + d_label) % len(label_names)
            self.update()

        self.prev_label = lambda: change_label(-1)
        self.next_label = lambda: change_label(1)

        self.cancel_toggle = False
        self.cancel_area = SelectedArea(label= self.current_label)

        self.autocontours = SelectedArea(label= self.current_label)
        
    def start_auto_segment(self):
        event_loop = asyncio.get_event_loop()
        (
            event_loop
                .run_in_executor(None, lambda: self.auto_segment())
                .add_done_callback(lambda future: self.update_pixmap())
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap().isNull():
            return
        painter = QPainter(self)
        status_texts = []
        #<-add status text here
        status_texts.append(f"Current Mode: {self.maskMode}")
        status_texts.append(f"Current Label: {label_names[self.current_label]}")
        cancel_text = "Cancel" if self.cancel_toggle else "Draw"
        status_texts.append(f"Cancel Mode: {cancel_text}")
        for i, status_text in enumerate(status_texts):
            # background
            font_metrics = painter.fontMetrics()
            text_bb = font_metrics.boundingRect(status_text)
            painter.setPen(QColor(255, 255, 255, 100))
            painter.drawRect(10, 10 + 10 * i, text_bb.width() + 10, text_bb.height())
            # text
            painter.setPen(QColor(0, 0, 0, 255))
            painter.drawText(10, 10 + 10 * i, status_text)

        for label, area in self.label_area_map.items():
            ramdomsrc = np.random.RandomState((label * label) * 0x4C1906 + (label * 0x5AC0DB))
            rc = QColor(ramdomsrc.randint(0, 255), ramdomsrc.randint(0, 255), ramdomsrc.randint(0, 255), 150)
            painter.setPen(rc)
            self.map_from_pixmap(area).draw(painter)
            fill_path = QPainterPath()
            self.map_from_pixmap(area).addToFillPath(fill_path)

            if self.current_label == label:
                fill_path.addPolygon(self.map_from_pixmap(self.polygon_cache))
                if not self.rectangle_cache.isNull():
                    painter.drawRect(self.map_from_pixmap(self.rectangle_cache))
                    fill_path.addRect(self.map_from_pixmap(self.rectangle_cache))

            rc1 = QColor(rc.red(), rc.green(), rc.blue(), 50)
            painter.fillPath(fill_path, rc1) 

        
    @overload
    def map_to_pixmap(self,point:QPoint)->QPoint:
        pass

    @overload
    def map_to_pixmap(self,point:QRect)->QRect:
        pass

    @overload
    def map_to_pixmap(self,point:QPolygonF)->QPolygonF:
        pass

    @overload
    def map_to_pixmap(self,point:'SelectedArea')->'SelectedArea':
        pass

    def map_to_pixmap(self,point):
        if self.pixmapscale_factor == 0:
            return QPoint(point.x(),point.y())
        
        if isinstance(point, QRect):
            return QRect(self.map_to_pixmap(point.topLeft()), self.map_to_pixmap(point.bottomRight()))
        
        if isinstance(point, QPolygonF):
            return QPolygonF([QPoint(p.x()/self.pixmapscale_factor,p.y()/self.pixmapscale_factor) for p in point])
        
        if isinstance(point, SelectedArea):
            return SelectedArea([self.map_to_pixmap(polygon) for polygon in point],label=point.label)
        
        return QPoint(point.x()/self.pixmapscale_factor,point.y()/self.pixmapscale_factor)
    
    @overload
    def map_from_pixmap(self,point:QPoint)->QPoint:
        pass

    @overload
    def map_from_pixmap(self,point:QRect)->QRect:
        pass

    @overload
    def map_from_pixmap(self,point:QPolygonF)->QPolygonF:
        pass

    @overload
    def map_from_pixmap(self,point:'SelectedArea')->'SelectedArea':
        pass
    
    def map_from_pixmap(self,point):
        if self.pixmapscale_factor == 0:
            return QPoint(point.x(),point.y())
        
        if isinstance(point, QRect):
            return QRect(self.map_from_pixmap(point.topLeft()), self.map_from_pixmap(point.bottomRight()))
        
        
        if isinstance(point, QPolygonF):
            return QPolygonF([QPoint(p.x()*self.pixmapscale_factor,p.y()*self.pixmapscale_factor) for p in point])
        
        if isinstance(point, SelectedArea):
            return SelectedArea([self.map_from_pixmap(polygon) for polygon in point],label=point.label)
        
        return QPoint(point.x()*self.pixmapscale_factor,point.y()*self.pixmapscale_factor)
            
    def mousePressEvent(self, event:QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.maskMode == "polygon":
                polygon = QPolygonF()
                polygon.append(self.map_to_pixmap(event.pos()))
                self.polygon_cache = polygon

            if self.maskMode == "rectangle":
                self.rectangle_cache = QRect(self.map_to_pixmap(event.pos()), self.map_to_pixmap(event.pos()))

        if event.button() == Qt.MouseButton.RightButton:
            self.clear_cache()
        self.update()
        
    def mouseMoveEvent(self, event:QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.maskMode == "polygon":
                self.polygon_cache.append(self.map_to_pixmap(event.pos()))

            if self.maskMode == "rectangle":
                self.rectangle_cache = QRect(self.rectangle_cache.topLeft(), self.map_to_pixmap(event.pos()))

            self.update()
            
    def mouseReleaseEvent(self, event:QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                area = self.cancel_area if self.cancel_toggle else self.selected_area()
                if self.maskMode == "polygon":
                    self.polygon_cache.append(self.map_to_pixmap(event.pos()))
                    area.add_polygon(self.polygon_cache)
                    self.polygon_cache = QPolygonF()

                if self.maskMode == "rectangle":
                    area.add_rect(QRect(self.rectangle_cache.topLeft(), self.map_to_pixmap(event.pos())))
                    self.rectangle_cache = QRect()

                if self.maskMode == "auto":
                    mousePos = self.map_to_pixmap(event.pos())
                    for poly in self.autocontours:
                        if poly.containsPoint(mousePos, Qt.FillRule.WindingFill):
                            area.add_polygon(poly)

                self.label_area_map[self.current_label] = self.shrink_area_to_area(self.selected_area(), self.cancel_area)
                self.cancel_area = SelectedArea(label= self.current_label)
                self.update_pixmap()
                self.update()

            except Exception as e:
                raise e
  
        elif event.button() == Qt.MouseButton.RightButton:
            mousePos = self.map_to_pixmap(event.pos())
            for poly in self.selected_area():
                if poly.containsPoint(mousePos, Qt.FillRule.WindingFill):
                    self.selected_area().remove_polygon(poly)
            self.update_pixmap()

    def clear_cache(self):
        self.polygon_cache = QPolygonF()
        self.rectangle_cache = QRect()
        self.cancel_area = SelectedArea(label=self.current_label,)
        
    def update_pixmap(self):

        roi = self.shrink_area_to_mask(self.selected_area())

        print("roi.shape",roi.shape)

        cv2.imwrite("test.png",roi)

        self.processed_pixmap=QPixmap("test.png")
        self.show_pixmap=QPixmap(self.original_pixmap.size())
        self.show_pixmap.fill(Qt.GlobalColor.white)

        
        painter=QPainter(self.show_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Darken)
        painter.setOpacity(1)
        painter.drawPixmap(0,0,self.original_pixmap)
        painter.setOpacity(0.5)
        painter.drawPixmap(0,0,self.processed_pixmap)
        painter.setOpacity(1)
        painter.end()

        self.setPixmap(self.show_pixmap)
        self.resizeEvent(QResizeEvent(self.size(),self.size()))

        self.update()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.label_area_map.clear()
            print("clear")
            self.setPixmap(self.original_pixmap)
            self.update()
        
        # mode switch
        if event.key() == Qt.Key.Key_P:
            self.maskMode = "polygon"
        if event.key() == Qt.Key.Key_R:
            self.maskMode = "rectangle"
        if event.key() == Qt.Key.Key_A:
            self.maskMode = "auto"
        print("maskMode",self.maskMode)

        if event.key() == Qt.Key.Key_C:
            self.cancel_toggle = not self.cancel_toggle
            self.cancel_area = SelectedArea(label= self.current_label)
        print("cancel_toggle",self.cancel_toggle)

        self.update_pixmap()

        return 

    def resizeEvent(self, arg__1: QResizeEvent) -> None:
        size_width =arg__1.size().width()
        self.setPixmap(
            self.show_pixmap.scaled(
                QSize(size_width, self.show_pixmap.height()+3000),
                  Qt.AspectRatioMode.KeepAspectRatio, 
                  Qt.TransformationMode.FastTransformation))
        
        self.pixmapscale_factor=self.pixmap().width()/(self.original_pixmap.width()+0.1)
            
        return super().resizeEvent(arg__1)
    
    def save_processed_image(self,path):
        print(f"save_processed_image to {path}")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        self.processed_pixmap.save(path)

    def save_yolo_seg(self,path):
        image_width=self.original_pixmap.width()
        image_height=self.original_pixmap.height()
        yolos = []
        for label, area in self.label_area_map.items():
            yolos.extend(area.to_yolo(image_width,image_height))
        with open(path,"w") as f:
            f.write("\n".join(yolos))

    def change_pic(self,pic_path:str):
        self.original_pixmap = QPixmap(pic_path)
        self.processed_pixmap = QPixmap(pic_path)
        self.show_pixmap = QPixmap(pic_path)
        
        self.setPixmap(self.show_pixmap)
        self.pixmapscale_factor=self.pixmap().width()/(self.original_pixmap.width()+0.1)
        self.update()

        image = cv2.imread(pic_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image

        self.clear_cache()
        self.label_area_map.clear()
        self.update_pixmap()
        self.start_auto_segment()
    
    def shrink_area_to_mask(self,area:'SelectedArea', area_reversed:'SelectedArea'=None):
        """Shrink the area to the where the contour of the mask is"""
        assert area is not None
        img = qpixmap_to_cvimage(self.original_pixmap)
        heith,width=img.shape[:2]

        mask=np.zeros((heith,width),dtype=np.uint8)
        #ptss=[[(int(p.x()*self.pixmapscale_factor),int(p.y()*self.pixmapscale_factor)) for p in polygon] for polygon in self.polygons]
        ptss=[[(int(p.x()),int(p.y())) for p in polygon] for polygon in area]
        
        ptss=[np.array(pts,dtype=np.int32)for pts in ptss]
        #cv2.fillPoly(mask,ptss,255)
        for pts in ptss:
            cv2.fillPoly(mask,[pts],255)

        if area_reversed is not None:
            ptss=[[(int(p.x()),int(p.y())) for p in polygon] for polygon in area_reversed]
            ptss=[np.array(pts,dtype=np.int32)for pts in ptss]
            for pts in ptss:
                cv2.fillPoly(mask,[pts],0)
        
        roi=cv2.bitwise_and(img,img,mask=mask)

        
        roi=cv2.cvtColor(roi,cv2.COLOR_RGBA2GRAY)
        _,thres=cv2.threshold(roi,white_thres,255,cv2.THRESH_BINARY_INV)
        #thres=cv2.bitwise_not(thres)
        thres=cv2.bitwise_and(thres,thres,mask=(mask))
        roi=thres
        return roi
    
    def shrink_area_to_area(self,area:'SelectedArea', area_reversed:'SelectedArea'=None):
        """Shrink the area to the where the contour of the mask is"""
        roi = self.shrink_area_to_mask(area, area_reversed)
        contours_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(np.where([cv2.contourArea(c) > 5 for c in contours_[0]]))
        contours = contours_[0]
        contours = [c for c in contours if cv2.contourArea(c) > 25]

        # contour = contours_[0][np.argmax([len(c) for c in contours_[0]])]
        # print("contour.shape",contour.shape)
        # polygon = QPolygonF([QPoint(x, y) for x, y in contour[:, 0]])
        # return polygon
        area = SelectedArea(label= self.current_label)
        for contour in contours:
            polygon = QPolygonF([QPoint(x, y) for x, y in contour[:, 0]])
            area.add_polygon(polygon)

        return area

    def auto_segment(self):
        image=qpixmap_to_cvimage(self.original_pixmap)
        if image is None:
            return
        segs = seg_cv2_yolo(image)
        for label, area in segs.items():
            print(label, area.area())
            if not label in self.label_area_map:
                self.label_area_map[label] = SelectedArea(label=label)
            self.label_area_map[label].merge(area)
            
        self.label_area_map[self.current_label] = self.shrink_area_to_area(self.selected_area(), self.cancel_area)
                

    async def boot(self):
        pass
    