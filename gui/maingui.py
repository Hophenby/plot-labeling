import sys,os,re
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton, 
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QScrollArea,
    QInputDialog,
    QTextEdit,
    )
from PySide6.QtCore import Slot, Qt, QSize, QPoint, QEvent
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QPixmap, 
    QResizeEvent,
    QPainter,
    QColor,
    QPolygonF,
    QMouseEvent,
    QPainterPath,
    QImage,
    )
from qasync import QEventLoop, QThreadExecutor, run, asyncSlot, asyncClose
import asyncio

from PIL import Image

from maskarea import MaskArea, label_names

indexed_folder = "indexed"
os.makedirs(indexed_folder,exist_ok=True)

origin_pattern = re.compile(r"\\(\d+)\\origin\.png$")
masked_pattern = re.compile(r"\\(\d+)\\masked\.png$")


def make_filename_legal(path:str):
    path.replace("\\","-")
    path.replace("/","-")
    path.replace(":","-")
    path.replace("*","-")
    path.replace("?","-")
    path.replace("\"","-")
    path.replace("<","-")
    path.replace(">","-")
    path.replace("|","-")
    return path

class MainGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main GUI")
        self.resize(400, 300)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_ = QVBoxLayout()
        self.central_widget.setLayout(self.layout_)

        self.pics=[] # {"path":str,"dir":str,"ind":int}
        self.current_pic_index=None

        self.maskarea = MaskArea()
        self.layout_.addWidget(self.maskarea)

        def parse_jumper():
            try:
                text = self.jumper.toPlainText()
                page = int(text)
                assert 0<=page<len(self.pics)
                self.current_pic_index = page-1
                self.change_pic()
            except:
                return None
            
        def refresh_jumper():
            self.jumper.setText(str(self.current_pic_index+1))

        label_layout = QHBoxLayout()
        prev_button = QPushButton("prev_label")
        next_button = QPushButton("next_label")
        label_info = QLabel("label_info")
        label_info.setFixedWidth(100)
        label_info.setFixedHeight(30)
        label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_info = label_info
        prev_button.clicked.connect(self.prev_label)
        next_button.clicked.connect(self.next_label)
        label_layout.addWidget(prev_button)
        label_layout.addWidget(label_info)
        label_layout.addWidget(next_button)
        self.layout_.addLayout(label_layout)


        self.prevnext=QHBoxLayout()
        self.prev_button = QPushButton("prev")
        self.next_button = QPushButton("next")
        self.jumper = QTextEdit()
        self.prev_button.clicked.connect(self.prev_pic)
        self.next_button.clicked.connect(self.next_pic)
        self.prev_button.clicked.connect(refresh_jumper)
        self.next_button.clicked.connect(refresh_jumper)
        self.jumper.textChanged.connect(parse_jumper)
        self.jumper.setFixedWidth(50)
        self.jumper.setFixedHeight(30)
        self.prevnext.addWidget(self.prev_button)
        self.prevnext.addWidget(self.jumper)
        self.prevnext.addWidget(self.next_button)
        self.layout_.addLayout(self.prevnext)
        
        self.jumptosave=QHBoxLayout()
        self.save_button = QPushButton("save")
        self.jumptosave.addWidget(self.save_button)

        self.layout_.addLayout(self.jumptosave)

 
    def change_pic(self):
        if self.current_pic_index is None:
            return
        self.maskarea.change_pic(self.pics[self.current_pic_index]["path"])
        try:
            self.save_button.clicked.disconnect()
        except:
            pass
        
        self.save_button.clicked.connect(
            lambda :self.maskarea.save_yolo_seg(
                path=os.path.join(
                    indexed_folder,
                    str(self.pics[self.current_pic_index]["ind"]),
                    f"yolo_seg.txt")
                    )
                )
        
        self.save_button.setText(f"save [{self.current_pic_index+1}/{len(self.pics)}]")
        self.label_info.setText(label_names[self.maskarea.current_label])

    def add_pic(self,pic:Image.Image):
        max_ind = max([pic["ind"] for pic in self.pics] + [-1])
        pic_path=os.path.join(indexed_folder,f"{max_ind+1}")
        os.makedirs(pic_path,exist_ok=True)
        pic.save(os.path.join(pic_path,"origin.png"))
        #self.pics.append(pic_path)
        def after_reload(x):
            self.current_pic_index=len(self.pics)-1
            self.jumper.setText(str(self.current_pic_index+1))
            self.change_pic()
        asyncio.ensure_future(self.load_pics()).add_done_callback(after_reload)
        #self.change_pic()

    def prev_pic(self):
        self.current_pic_index = self.current_pic_index or 0
        self.current_pic_index-=1
        self.current_pic_index%=len(self.pics)
        self.change_pic()

    def next_pic(self):
        self.current_pic_index = self.current_pic_index if self.current_pic_index is not None else -1
        self.current_pic_index+=1
        self.current_pic_index%=len(self.pics)
        self.change_pic()

    def prev_label(self):
        self.maskarea.prev_label()
        label_info = label_names[self.maskarea.current_label]
        self.label_info.setText(label_info)

    def next_label(self):
        self.maskarea.next_label()
        label_info = label_names[self.maskarea.current_label]
        self.label_info.setText(label_info)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        #print(event.keyCombination().toCombined(), Qt.Key.Key_Paste)
        if (event.key() == Qt.Key.Key_V and 
            event.modifiers() == Qt.KeyboardModifier.ControlModifier and
            QApplication.clipboard().mimeData().hasImage()):
            QApplication.clipboard().mimeData().imageData().save("clipboard.png")
            self.add_pic(Image.open("clipboard.png"))
        self.maskarea.keyReleaseEvent(event)
        return super().keyReleaseEvent(event)
        
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        return super().dragEnterEvent(event)
    
    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            self.add_pic(Image.open(event.mimeData().urls()[0].toLocalFile()))
            event.accept()
        return super().dropEvent(event)

    @Slot()
    def start_background_task(self):
        #self.label.setText("Running background task...")
        self.save_button.setEnabled(False)
        asyncio.ensure_future(self.background_task())

    async def background_task(self):
        await asyncio.sleep(5)
        #self.label.setText("Background task completed.")
        self.save_button.setEnabled(True)

    async def load_pics(self):
        self.pics=[]
        for dir,folders,files in os.walk(indexed_folder):
            for file in files:
                file_path=os.path.join(dir,file)
                if origin_pattern.search(file_path):
                    print(f"Loading {file_path}")
                    dir1=os.path.split(dir)[1]
                    self.pics.append({
                        "path":os.path.join(dir,file),
                        "dir":dir1,
                        "ind":int(origin_pattern.search(file_path).group(1))
                    })
        self.pics.sort(key=lambda x:x["ind"])

    
    async def boot(self):
        await self.load_pics()
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)

    event_loop = QEventLoop(app)
    asyncio.set_event_loop(event_loop)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)
    
    main_window = MainGui()
    main_window.show()

    event_loop.create_task(main_window.boot())
    event_loop.run_until_complete(app_close_event.wait())
    event_loop.close()