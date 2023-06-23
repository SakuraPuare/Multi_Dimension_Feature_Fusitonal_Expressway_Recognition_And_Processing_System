# coding=utf-8
import json
import pathlib
import random
import re
import sys
import time
from threading import Thread
from typing import Union, List, Any

import cv2
import numpy as np
import torch
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import QTableWidgetItem, QLineEdit, QPushButton, QDialog, QHBoxLayout, QLabel, QWidget
from torch import nn
from torch.backends import cudnn

from models.common import Conv
from models.experimental import Ensemble
from ui import barrier as ui_barrier
from ui import config as ui_config
from ui import main as ui_main
from ui import source as ui_source
from utils.datasets import letterbox
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def clear_str(l: List[Any]):
    t = []
    for i in l:
        if i is None:
            t.append(None)
        elif isinstance(i, str):
            t.append(re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=i))
        else:
            t.append(i)
    return t


def attempt_load(weights, device=None) -> Ensemble:
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=device)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


class Render(QLabel):
    class Stream:
        def __init__(self, sources, img_size=640, stride=32):
            self.img = []

            self.sources = sources  # clean source names for later
            self.img_size = img_size
            self.stride = stride

            self._stop = False
            # Start the thread to read frames from the video stream
            url = eval(self.sources) if self.sources.isnumeric() else self.sources

            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                print(f'Failed to open {url}')
                raise IOError
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) % 100

            ret, self.img = self.cap.read()  # guarantee first frame
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')

            self.thread = Thread(target=self.update, args=([self.cap]), daemon=True)
            self.thread.start()
            # check for common shapes
            s = np.stack(letterbox(self.img, self.img_size, stride=self.stride)[0].shape)  # shapes
            self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

        def update(self, cap):
            # Read next stream frame in a daemon thread
            n = 0
            # while cap and thread not to stop
            while cap.isOpened() and self.thread.is_alive() and not self._stop:
                n += 1
                cap.grab()
                if n == 2:  # read every 4th frame
                    success, im = cap.retrieve()
                    self.img = im if success else self.img * 0
                    n = 0
                time.sleep(1 / self.fps)  # wait time

        def stop(self):
            self.cap.release()
            # send stop signal to thread
            self._stop = True
            self.thread.join()

        def __iter__(self):
            self.count = -1
            return self

        def __next__(self):
            self.count += 1
            img0 = self.img.copy()
            if cv2.waitKey(1) == ord('q'):  # q to quit
                cv2.destroyAllWindows()
                raise StopIteration
            # Letterbox
            img = letterbox(img0, self.img_size, auto=self.rect, stride=self.stride)[0]
            # Stack
            img = np.stack(img)
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)
            return img, img0

        def __len__(self):
            return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

    class Render:
        def __init__(self, parent, video_url):
            self.parent = parent

            self._stop = False
            self.video_url = video_url

            self.stream = Render.Stream(sources=self.video_url)
            self.thread = Thread(target=self.run, daemon=True)

        def run(self):
            for _, img0 in self.stream:
                # stop the thread if stop signal received
                if self._stop:
                    break

                rgb_image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.parent.label.setPixmap(QPixmap.fromImage(p))

        def start(self):
            self.thread.start()

        def stop(self):
            self._stop = True
            self.stream.stop()
            self.thread.join()

    class Detect:
        def __init__(self, parent, video_url, img_size=640, iou_threshold=0.45, conf_threshold=0.25):
            self.parent = parent
            self._stop = False

            self.video_url = video_url
            self.img_size = img_size
            self.iou_threshold = iou_threshold
            self.conf_threshold = conf_threshold

            self.half = device.type != 'cpu'
            self.stride = int(model.stride.max())  # model stride
            self.imgsz = check_img_size(self.img_size, s=self.stride)

            if self.half:
                model.half()  # to FP16
            check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference

            self.stream = Render.Stream(sources=self.video_url, img_size=self.img_size, stride=self.stride)

            self.names = model.module.names if hasattr(model, 'module') else model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

            if device.type != 'cpu':
                model(
                    torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(model.parameters())))  # run once
            self.old_img_w = self.old_img_h = self.imgsz
            self.old_img_b = 1

            self.thread = Thread(target=self.run, daemon=True)

        def run(self):
            for img, im0s in self.stream:
                if self._stop:
                    break

                img = torch.from_numpy(img).to(device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (
                        self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[
                    3]):
                    self.old_img_b = img.shape[0]
                    self.old_img_h = img.shape[2]
                    self.old_img_w = img.shape[3]
                    for i in range(3):
                        model(img)[0]

                # Inference
                with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                    pred = model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    im0 = im0s.copy()

                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

                rgb_image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.parent.label.setPixmap(QPixmap.fromImage(p))

        def start(self):
            self.thread.start()

        def stop(self):
            self._stop = True
            self.stream.stop()
            self.thread.join()

    def __init__(self, parent):
        super().__init__(parent.widget)
        self.parent = parent

        self.label = self.parent.VideoLabel

        self.sync()

    def get_image(self) -> QPixmap | None:
        if self.video_url == '':
            return None
        return self.label.pixmap()

    def sync(self):
        self.video_url = self.parent.video_url
        self.iou_threshold = self.parent.iou_threshold
        self.conf_threshold = self.parent.conf_threshold
        self.img_size = self.parent.img_size
        self.is_detect = self.parent.is_detect

    def reload(self):
        self.sync()

        if self.video_url == '':
            return

        self.release()

        if self.is_detect:
            self.detect = self.Detect(self, self.video_url, self.img_size, self.iou_threshold, self.conf_threshold)
        else:
            self.detect = self.Render(self, self.video_url)
        self.detect.start()

    def release(self):
        if hasattr(self, 'detect') and self.detect and self.detect.thread.is_alive():
            self.detect.stop()

    def close(self):
        self.release()
        # clear image
        self.label.clear()


class SettingList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'<SettingList {super().__repr__()}>'

    @property
    def increase_ids(self):
        return (max(self).ids if len(self) else 0) + 1

    def swap(self, index1: int, index2: int):
        t = self[index1]
        self[index1] = self[index2]
        self[index2] = t

    def up_move(self, index: int):
        if index == 0:
            return
        self.swap(index, index - 1)

    def down_move(self, index: int):
        if index == len(self) - 1:
            return
        self.swap(index, index + 1)


class SettingItem:
    __keys__ = ['ids', 'name', 'url']

    def __init__(self, ids: int = None, name: str = None, url: Union[str, int, None] = None, types: str = None) -> None:
        self.ids = ids
        self.name = name
        self.url = url

    def __dict__(self):
        ret = {}
        for key in self.__keys__:
            ret[key] = getattr(self, key)
        return ret

    def __repr__(self):
        return f'<SettingItem ids={self.ids}, name={self.name}, url={self.url}>'

    def __gt__(self, other):
        return self.ids > other.ids


class ConfigWindow(ui_config.Ui_Dialog):
    def __init__(self, parent=None):
        self.parent = parent
        self.dialog = QtWidgets.QDialog(parent.widget)
        super().__init__()
        self.setupUi(self.dialog)

        self.SaveButton.clicked.connect(self.save_setting)
        self.CancelButton.clicked.connect(self.dialog.close)

        self.load_setting()

    def load_setting(self):
        # set text in line edit
        self.IoUEdit.setText(str(self.parent.iou_threshold))
        self.ConfEdit.setText(str(self.parent.conf_threshold))
        self.SizeEdit.setText(str(self.parent.img_size))

    def save_setting(self):
        self.parent.iou_threshold = float(self.IoUEdit.text())
        self.parent.conf_threshold = float(self.ConfEdit.text())
        self.parent.img_size = float(self.SizeEdit.text())
        self.parent.save_config()
        self.dialog.close()


class SourceWindows(ui_source.Ui_dialog):
    def __init__(self, parent=None):
        self.parent = parent
        self.setting_list = SettingList()
        super().__init__()
        self.dialog = QtWidgets.QDialog(parent.widget)
        self.setupUi(self.dialog)

        self.tableWidget.horizontalHeader().setStretchLastSection(True)

        self.AddSourceButton.clicked.connect(self.add_setting)
        self.EditSourceButton.clicked.connect(self.edit_setting)
        self.DelectSouceButton.clicked.connect(self.delete_setting)
        self.UpMoveButton.clicked.connect(self.up_move_setting)
        self.DownMoveButton.clicked.connect(self.down_move_setting)
        self.FlushSourceButton.clicked.connect(self.load_setting)

        self.SaveButton.clicked.connect(self.save_setting)
        self.SelectButton.clicked.connect(self.select_setting)
        self.CancelButtoon.clicked.connect(self.dialog.close)
        self.CloseSourceButton.clicked.connect(self.close_setting)

        self.load_setting()

    def add_setting(self):
        def confirm():
            name = name_edit.text()
            url = url_edit.text()
            if url == '':
                return
            setting = SettingItem(ids=self.setting_list.increase_ids, name=name, url=url)
            self.setting_list.append(setting)
            self.flush_setting()
            # self.save_to_config()
            add_widget.close()

        name_edit = QLineEdit(self.dialog)
        url_edit = QLineEdit(self.dialog)

        confirm_button = QPushButton('确定', self.dialog)
        cancel_button = QPushButton('取消', self.dialog)

        add_widget = QDialog(self.dialog)
        add_widget.setWindowTitle('添加媒体源')

        layout = QHBoxLayout(add_widget)
        layout.addWidget(QLabel('名称:', self.dialog))
        layout.addWidget(name_edit)
        layout.addWidget(QLabel('地址:', self.dialog))
        layout.addWidget(url_edit)
        layout.addWidget(confirm_button)
        layout.addWidget(cancel_button)

        confirm_button.clicked.connect(confirm)
        cancel_button.clicked.connect(add_widget.close)

        add_widget.show()

    def edit_setting(self):
        def confirm():
            setting.name = name_edit.text()
            setting.url = url_edit.text()
            setting.types = types_edit.text()

            row = self.tableWidget.currentRow()
            self.setting_list[row] = setting

            self.flush_setting()
            # self.save_to_config()
            edit_widget.close()

        row = self.tableWidget.currentRow()
        if row != -1:
            setting: SettingItem = self.setting_list[row]

            name_edit = QLineEdit(self.dialog)
            url_edit = QLineEdit(self.dialog)
            types_edit = QLineEdit(self.dialog)

            confirm_button = QPushButton('确定', self.dialog)
            cancel_button = QPushButton('取消', self.dialog)

            edit_widget = QDialog(self.dialog)
            edit_widget.setWindowTitle('编辑媒体源')
            layout = QHBoxLayout()
            layout.addWidget(QLabel('名称:', self.dialog))
            layout.addWidget(name_edit)
            layout.addWidget(QLabel('地址:', self.dialog))
            layout.addWidget(url_edit)

            layout.addWidget(confirm_button)
            layout.addWidget(cancel_button)
            edit_widget.setLayout(layout)

            name_edit.setText(setting.name)
            url_edit.setText(str(setting.url))

            confirm_button.clicked.connect(confirm)
            cancel_button.clicked.connect(edit_widget.close)

            edit_widget.show()

    def save_setting(self):
        with open(source_path, 'w', encoding='utf-8') as f:
            json.dump(self.setting_list, f, default=lambda x: x.__dict__(), ensure_ascii=False, indent=4)
        self.dialog.close()

    def load_setting(self):
        if not source_path.exists():
            self.save_setting()
        with open(source_path, 'r', encoding='u8') as f:
            self.setting_list = SettingList(json.load(f, object_hook=lambda x: SettingItem(**x)))
        self.flush_setting()

    def flush_setting(self):
        self.tableWidget.setRowCount(len(self.setting_list))
        for i, setting in enumerate(self.setting_list):
            for j, key in enumerate(SettingItem.__keys__):
                item = QTableWidgetItem(str(getattr(setting, key)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tableWidget.setItem(i, j, item)

    def delete_setting(self):
        row = self.tableWidget.currentRow()
        if row != -1:
            self.setting_list.pop(row)
            self.flush_setting()

    def up_move_setting(self):
        row = self.tableWidget.currentRow()
        if row != -1:
            self.setting_list.up_move(row)
            row -= 1
            self.flush_setting()
            self.tableWidget.selectRow(row)

    def down_move_setting(self):
        row = self.tableWidget.currentRow()
        if row != -1:
            self.setting_list.down_move(row)
            row += 1
            self.flush_setting()
            self.tableWidget.selectRow(row)

    def select_setting(self):
        row = self.tableWidget.currentRow()
        if row != -1:
            self.parent.video_url = self.setting_list[row].url
            self.dialog.close()

    def close_setting(self):
        self.parent.video_url = ''
        self.dialog.close()


class BarrierWindows(ui_barrier.Ui_widget):
    class DrawWidget(QWidget):
        def __init__(self, parent, pixmap):
            super().__init__(parent)
            self.parent = parent
            self.pixmap = pixmap

            self.last_point = QtCore.QPoint()
            self._is_draw = False

            # from pixmap to image
            self.image = self.pixmap.toImage()
            self.image = self.image.scaled(640, 480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.painter = QtGui.QPainter(self.image)

            self.painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 10, QtCore.Qt.PenStyle.SolidLine,
                                           QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
            self.painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0), QtCore.Qt.BrushStyle.SolidPattern))
            self.painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            self.painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 0, self.image)

        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # get pointer position relative to the widget
                self.last_point = event.pos()
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                # draw line
                self.painter.drawLine(self.last_point, event.pos())
                self.update()



    def __init__(self, parent):
        self.parent = parent
        super().__init__()

        self._state = False

        self.widget = QtWidgets.QWidget()
        self.setupUi(self.widget)

        self.init()

    def get_image(self) -> QPixmap | None:
        return self.parent.render.get_image()

    def load_image(self):
        pixmap = self.get_image()
        if pixmap is None:
            # render text
            pixmap = QtGui.QPixmap(640, 480)
            pixmap.fill(QtGui.QColor(255, 255, 255))
            # fit image
            pixmap = pixmap.scaled(640, 480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            # write text
            painter = QtGui.QPainter(pixmap)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1, QtCore.Qt.PenStyle.SolidLine,
                                      QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0), QtCore.Qt.BrushStyle.SolidPattern))
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            painter.drawText(0, 0, 640, 480, QtCore.Qt.AlignmentFlag.AlignCenter, 'No Image')
            painter.end()
            return pixmap
        return pixmap

    def show_text(self, text: str, font_size: int = 20):
        # set text
        self.ImageLabel.setText(text)
        self.ImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ImageLabel.setWordWrap(True)
        # font
        font = QFont()
        font.setPointSize(font_size)
        self.ImageLabel.setFont(font)

    def change(self):
        if self._state:
            self._state = True
            self.horizontalLayout.replaceWidget(self.ImageLabel, self.draw)
        else:
            self._state = False
            self.horizontalLayout.replaceWidget(self.draw, self.ImageLabel)

    def set_button(self, state: bool):
        self.ArrorTool.setEnabled(state)
        self.PencelTool.setEnabled(state)
        self.EraserTool.setEnabled(state)

        self.UndoButton.setEnabled(state)
        self.SaveButton.setEnabled(state)

    def disable(self):
        self.set_button(False)

    def enable(self):
        self.set_button(True)

    def init(self):
        self.draw = self.DrawWidget(self.widget, self.load_image())
        self.draw.resize(self.ImageLabel.size())

        self.init_paint()

        self.widget.show()

    def init_paint(self):
        pixmap = self.get_image()
        if pixmap is None:
            self.disable()
            self.ImageLabel.setText('No Image')
        else:
            self.enable()
            self.change()


class MainWindows(ui_main.Ui_Form):
    _video_url = ''
    _iou_threshold = 0.25
    _conf_threshold = 0.45
    _img_size = 640
    _is_detect = True

    @property
    def video_url(self):
        return self._video_url

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @property
    def conf_threshold(self):
        return self._conf_threshold

    @property
    def img_size(self):
        return self._img_size

    @property
    def is_detect(self):
        return self._is_detect

    @video_url.setter
    def video_url(self, url):
        print(url)
        self._video_url = url
        if url == '':
            self.render.close()
            self.show_text('请设置视频源')
        else:
            self.render.reload()

    @iou_threshold.setter
    def iou_threshold(self, value):
        self._iou_threshold = value

    @conf_threshold.setter
    def conf_threshold(self, value):
        self._conf_threshold = value

    @img_size.setter
    def img_size(self, value):
        self._img_size = value
        self.render.reload()

    @is_detect.setter
    def is_detect(self, value):
        self._is_detect = value
        self.render.reload()

    def __init__(self):
        super().__init__()
        self.widget = QtWidgets.QWidget()
        self.setupUi(self.widget)

        self.render = Render(self)

        self.show_text('请设置视频源')
        self.init()

        self._video_url = ''

    def show_text(self, text, font_size=40):
        # set text
        self.VideoLabel.setText(text)
        self.VideoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.VideoLabel.setWordWrap(True)
        # font
        font = QFont()
        font.setPointSize(font_size)
        self.VideoLabel.setFont(font)

    def init(self):
        self.load_config()

        self.sourceWindow = SourceWindows(self)
        self.settingWindow = ConfigWindow(self)

        # click to show source window
        self.SourceButton.clicked.connect(self.sourceWindow.dialog.show)
        self.thresholdButton.clicked.connect(self.settingWindow.dialog.show)
        self.barrierButton.clicked.connect(self.reload_barrier)

    def reload_barrier(self):
        self.barrierWindow = BarrierWindows(self)

    def save_config(self):
        with open(setting_path, 'w', encoding='utf-8') as f:
            json.dump({
                'iou_threshold': self.iou_threshold,
                'conf_threshold': self.conf_threshold,
                'img_size': self.img_size,
            }, f, ensure_ascii=False, indent=4)

    def load_config(self):
        if not setting_path.exists():
            self.save_config()
        with open(setting_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self._iou_threshold = config.get('iou_threshold', self._iou_threshold)
            self._conf_threshold = config.get('conf_threshold', self._conf_threshold)
            self._img_size = config.get('img_size', self._img_size)


if __name__ == "__main__":
    source_path = pathlib.Path('sources.json')
    setting_path = pathlib.Path('settings.json')
    device = select_device('0')
    model = attempt_load('yolov7.pt', device)
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindows()
    ui.widget.show()
    sys.exit(app.exec())
