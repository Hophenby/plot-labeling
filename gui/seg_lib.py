
from ultralytics import YOLO
import ultralytics.engine.results as yolo_results
import cv2
from polygon_lib import SelectedArea
from PySide6.QtCore import QPointF, QPoint
from PySide6.QtGui import QPolygonF
label_names = {0:"line",
               1:"bar",}

# Load a model
yolomodel = YOLO("models/n/20241105.pt")  # load a custom model

def seg_cv2_yolo(img:cv2.typing.MatLike) -> dict[int,SelectedArea]:
    """Segmentation using YOLO"""
    orig_shape = img.shape
    results: list[yolo_results.Results] = yolomodel(img)
    result = results[0] # we only have one image

    segs = result.to_df()

    masks = result.masks.data.numpy()

    returning = {label_class:SelectedArea(label=label_class) for label_class in label_names.keys()}
    for i,seg in segs.iterrows():
        area = returning[seg["class"]]
        
        seg_dict = seg["segments"]
        xs = seg_dict["x"]
        ys = seg_dict["y"]
        
        msk = masks[i]
        
        # Ensure the mask is in the correct format
        msk_uint8 = (msk * 255).astype('uint8')
        msk_uint8 = cv2.resize(msk_uint8, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_AREA)
        contours, _ = cv2.findContours(msk_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygon = QPolygonF()
        for contour in contours:
            polygon = QPolygonF([QPoint(x, y) for x, y in contour[:, 0]])
            area.add_polygon(polygon)
    return returning        

