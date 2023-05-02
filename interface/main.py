# coding=utf-8
import json
import pathlib
import random
import re
import sys
import time
from typing import Union, Tuple, Any, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtCore import (QThread, pyqtSignal, pyqtSlot, Qt)
from PyQt6.QtGui import (QImage, QPixmap, QIcon)
from PyQt6.QtWidgets import (QPushButton, QGridLayout, QLabel, QWidget, QMessageBox, QLineEdit, QDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QAbstractItemView, QSizePolicy,
                             QListWidget, QVBoxLayout,
                             QApplication)

from models.common import Conv
from models.experimental import Ensemble
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

settings_path = pathlib.Path('settings.json')


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


class SettingList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'<SettingList {super().__repr__()}>'

    @property
    def increase_ids(self):
        return (max(self).ids if len(self) else 0) + 1

    def up_move(self, index: int):
        if index == 0:
            return
        t = self[index]
        self[index] = self[index - 1]
        self[index - 1] = t

    def down_move(self, index: int):
        if index == len(self) - 1:
            return
        t = self[index]
        self[index] = self[index + 1]
        self[index + 1] = t


class SettingItem:
    __keys__ = ['ids', 'name', 'url', 'types']
    __table_header__ = __keys__

    def __init__(self, ids: int = None, name: str = None, url: Union[str, int, None] = None, types: str = None) -> None:
        self.ids = ids
        self.name = name

        self.url, self.types = self.load_url(url, types)

    @staticmethod
    def load_url(url, types: str):
        if types is None:
            types = str(type(url))
        elif types == "<class 'int'>":
            url = int(url)
        elif types == "<class 'str'>":
            url = str(url)
        elif types == "<class 'NoneType'>":
            url = None
        else:
            raise TypeError(f'unknown type {types}')
        return url, types

    def __dict__(self):
        return {
            'ids': self.ids,
            'name': self.name,
            'url': self.url,
            'types': self.types
        }

    def __repr__(self):
        return f'<SettingItem ids={self.ids}, name={self.name}, url={self.url}>'

    def __gt__(self, other):
        return self.ids > other.ids


class SettingWidget(QDialog):
    def __init__(self, parent, widget_size: Tuple[int, int] = (640, 480)):
        super().__init__(parent)
        self.parent = parent
        self.setting_table = None

        self.setting_list: SettingList[SettingItem] = parent.setting_list
        self.title = '视频源设置'

        self.h_layout = QHBoxLayout()
        self.grid = QGridLayout(self)
        self.grid.addLayout(self.h_layout, 0, 0, 1, 1)

        self.resize(*widget_size)
        parent.center(self)

        self.init()

    def flush(self):
        self.setting_table.setRowCount(len(self.setting_list))
        for i, setting in enumerate(self.setting_list):
            for j, key in enumerate(SettingItem.__keys__):
                item = QTableWidgetItem(str(getattr(setting, key)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setting_table.setItem(i, j, item)

    def save_to_config(self):
        with open(pathlib.Path(settings_path), 'w', encoding='utf-8') as f:
            json.dump(self.setting_list, f, default=lambda x: x.__dict__(), ensure_ascii=False, indent=4)

    def add_setting(self):
        def confirm():
            name = name_edit.text()
            url = url_edit.text()
            setting = SettingItem(ids=self.setting_list.increase_ids, name=name, url=url)
            self.setting_list.append(setting)
            self.flush()
            # self.save_to_config()
            add_widget.close()

        name_edit = QLineEdit(self)
        url_edit = QLineEdit(self)

        confirm_button = QPushButton('确定', self)
        cancel_button = QPushButton('取消', self)

        add_widget = QDialog(self)
        add_widget.setWindowTitle('添加媒体源')

        layout = QHBoxLayout(add_widget)
        layout.addWidget(QLabel('名称:', self))
        layout.addWidget(name_edit)
        layout.addWidget(QLabel('地址:', self))
        layout.addWidget(url_edit)
        layout.addWidget(confirm_button)
        layout.addWidget(cancel_button)

        confirm_button.clicked.connect(confirm)
        cancel_button.clicked.connect(add_widget.close)

        add_widget.show()

    def edit_setting(self):
        def confirm():
            types = types_edit.text()
            setting.name = name_edit.text()
            setting.url, _ = setting.load_url(url_edit.text(), types)

            self.flush()
            # self.save_to_config()
            edit_widget.close()

        row = self.setting_table.currentRow()
        if row != -1:
            setting: SettingItem = self.setting_list[row]

            name_edit = QLineEdit(self)
            url_edit = QLineEdit(self)
            types_edit = QLineEdit(self)

            confirm_button = QPushButton('确定', self)
            cancel_button = QPushButton('取消', self)

            edit_widget = QDialog(self)
            edit_widget.setWindowTitle('编辑媒体源')
            layout = QHBoxLayout()
            layout.addWidget(QLabel('名称:', self))
            layout.addWidget(name_edit)
            layout.addWidget(QLabel('地址:', self))
            layout.addWidget(url_edit)
            layout.addWidget(QLabel('类型:', self))
            layout.addWidget(types_edit)

            layout.addWidget(confirm_button)
            layout.addWidget(cancel_button)
            edit_widget.setLayout(layout)

            name_edit.setText(setting.name)
            url_edit.setText(str(setting.url))
            types_edit.setText(str(setting.types))

            confirm_button.clicked.connect(confirm)
            cancel_button.clicked.connect(edit_widget.close)

            edit_widget.show()

    def delete_setting(self):
        row = self.setting_table.currentRow()
        if row != -1:
            self.setting_list.pop(row)
            self.flush()
            # self.save_to_config()

    def up_move_setting(self):
        row = self.setting_table.currentRow()
        if row != -1:
            self.setting_list.up_move(row)
            row -= 1
            self.flush()
            # self.save_to_config()

    def down_move_setting(self):
        row = self.setting_table.currentRow()
        if row != -1:
            self.setting_list.down_move(row)
            row += 1
            self.flush()

            # self.save_to_config()

    def select_setting(self):
        row = self.setting_table.currentRow()
        if row != -1:
            self.parent.url = self.setting_list[row].url
            self.close()

    def save_setting(self):
        notice = QMessageBox(QMessageBox.Icon.Information, '提示', '是否保存设置？',
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
        if notice.exec() == QMessageBox.StandardButton.Yes:
            self.save_to_config()

    def close_setting(self):
        self.parent.url = None
        self.close()

    def init_table(self):
        self.setting_table = QTableWidget(self)

        self.h_layout.addWidget(self.setting_table)

        self.setting_table.setColumnCount(len(SettingItem.__table_header__))
        self.setting_table.setRowCount(len(self.setting_list))
        self.setting_table.setHorizontalHeaderLabels(SettingItem.__table_header__)
        self.setting_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setting_table.setMinimumSize(550, 400)

        self.setting_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.setting_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setting_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setting_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for i, setting in enumerate(self.setting_list):
            for j, key in enumerate(SettingItem.__keys__):
                item = QTableWidgetItem(str(getattr(setting, key)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setting_table.setItem(i, j, item)

    def init_button(self):
        self.button_flush = QPushButton('刷新', self)
        self.button_turn_off = QPushButton('关闭', self)
        self.button_add = QPushButton('添加', self)
        self.button_edit = QPushButton('编辑', self)
        self.button_delete = QPushButton('删除', self)
        self.button_up_move = QPushButton('上移', self)
        self.button_down_move = QPushButton('下移', self)
        self.button_select = QPushButton('选择', self)
        self.button_save = QPushButton('保存', self)
        self.button_cancel = QPushButton('取消', self)

        v_layout = QVBoxLayout()

        v_layout.addWidget(self.button_flush)
        v_layout.addWidget(self.button_turn_off)
        v_layout.addWidget(self.button_add)
        v_layout.addWidget(self.button_edit)
        v_layout.addWidget(self.button_delete)
        v_layout.addWidget(self.button_up_move)
        v_layout.addWidget(self.button_down_move)

        self.h_layout.addLayout(v_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.button_select)
        h_layout.addWidget(self.button_save)
        h_layout.addWidget(self.button_cancel)

        self.grid.addLayout(h_layout, 1, 0, 1, 2)

        self.button_flush.clicked.connect(self.flush)
        self.button_turn_off.clicked.connect(self.close_setting)
        self.button_add.clicked.connect(self.add_setting)
        self.button_edit.clicked.connect(self.edit_setting)
        self.button_delete.clicked.connect(self.delete_setting)
        self.button_up_move.clicked.connect(self.up_move_setting)
        self.button_down_move.clicked.connect(self.down_move_setting)

        self.button_select.clicked.connect(self.select_setting)
        self.button_save.clicked.connect(self.save_setting)
        self.button_cancel.clicked.connect(self.close)

        self.setting_table.cellDoubleClicked.connect(self.select_setting)

    def init(self):
        self.setWindowTitle(self.title)
        self.init_table()
        self.init_button()
        self.show()


class StreamDetectorWidget(QLabel):
    @pyqtSlot(QImage)
    def set_frame(self, image):
        self.setPixmap(QPixmap.fromImage(image))

    class DataStream(QLabel):
        class FrameThread(QThread):
            def __init__(self, parent, index, cap):
                super().__init__(parent)
                self.parent = parent
                self.index = index
                self.cap = cap

            def run(self):
                # Read next stream frame in a daemon thread
                n = 0
                while self.cap.isOpened() and not self.isInterruptionRequested():
                    n += 1
                    self.cap.grab()  # read every 4th frames
                    if n == 5:
                        n = 0
                        success, im = self.cap.retrieve()
                        self.parent.imgs[self.index] = im if success else self.parent.imgs[self.index] * 0
                    time.sleep(1 / self.parent.fps)  # wait time

            def stop(self):
                self.requestInterruption()
                self.wait()

        def __init__(self, parent):
            super().__init__(parent)
            self.parent = parent

            self.img_size = self.parent.img_size
            self.stride = self.parent.stride
            self.url = self.parent.url

            sources = [self.url]

            n = len(sources)
            self.imgs = [None] * n

            self.sources = clear_str(sources)

            for i, s in enumerate(sources):
                # Start the thread to read frames from the video stream
                print(f'{i + 1}/{n}: {s}... ', end='')
                url = eval(s) if s.isnumeric() else s
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print('Failed to open %s' % s)
                    return

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) % 100 or 24

                _, self.imgs[i] = cap.read()  # guarantee first frame

                self.frame_thread = self.FrameThread(self, index=i, cap=cap)
                print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
                self.frame_thread.start()

            print('')  # newline

            s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
            self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

        def __iter__(self):
            self.count = -1
            return self

        def __next__(self):
            self.count += 1
            img0 = self.imgs.copy()

            # Letterbox
            img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)

            return self.sources, img, img0

    class DetectThread(QThread):
        update_frame = pyqtSignal(QImage, name='update_frame')

        def __init__(self, parent):
            super().__init__(parent)
            self.parent = parent

            self.names = model.module.names if hasattr(model, 'module') else model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        def run(self):
            for path, img, im0s in self.parent.dataset:
                if self.isInterruptionRequested():
                    return

                img = torch.from_numpy(img).to(self.parent.device)
                img = img.half() if self.parent.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                    pred = model(img, augment=self.parent.augment)[0]

                pred = non_max_suppression(pred, self.parent.conf_threshold, self.parent.iou_threshold,
                                           classes=self.parent.classes, agnostic=self.parent.agnostic_nms)

                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.parent.dataset.count

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    # print(f'{s}Done. ')

                    rgb_image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.update_frame.emit(p)

        def stop(self):
            self.requestInterruption()
            self.wait()

    def __init__(self, parent, widget_size: Tuple[int, int] = (640, 480)):
        super().__init__(parent)
        self.parent = parent

        self.img_size = 640
        self.stride = int(model.stride.max())
        self.conf_threshold = self.parent.conf_threshold
        self.iou_threshold = self.parent.iou_threshold
        self.url = self.parent.url
        self.agnostic_nms = False
        self.classes = None

        self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        self.augment = False

        self.widget_size = widget_size

        # pdb.set_trace()
        if self.url is not None:
            self.dataset = self.DataStream(self)
            self.init_detector()
        self.init_ui()

    def init_detector(self):
        if self.half != 'cpu':
            model.half()

        self.detect_thread = self.DetectThread(self)
        self.detect_thread.update_frame.connect(self.set_frame)
        self.detect_thread.start()

    def init_ui(self):
        if self.url is None:
            # show text center
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setText('未设置视频源')
            self.font = self.font()
            self.font.setPointSize(40)
            self.setFont(self.font)

        self.parent.h2_layout.addWidget(self)

        self.resize(*self.widget_size)


class StreamWidget(QLabel):
    class FrameThread(QThread):
        update_frame = pyqtSignal(QImage, name='update_frame')

        def __init__(self, parent):
            super().__init__(parent)

            self.fps = 0
            self.height = 0
            self.width = 0

            self.url = parent.url

        def run(self):
            if self.url is None:
                return
            print(f'cap is opening `{self.url}` with no interruption.')
            cap = cv2.VideoCapture(self.url)
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100 or 30
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            print(f'fps: {self.fps} height: {self.height} width: {self.width}')
            while cap.isOpened() and not self.isInterruptionRequested():
                ret, frame = cap.read()
                if ret:
                    # https://stackoverflow.com/a/55468544/6622587
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.update_frame.emit(p)
                time.sleep(1 / self.fps)
            print('cap is closed')

        def stop(self):
            self.requestInterruption()
            self.wait()

    def __init__(self, parent, widget_size: Tuple[int, int] = (640, 480)):
        super().__init__(parent)
        self.parent = parent
        self.url = parent.url

        if self.url is None:
            # show text center
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setText('未设置视频源')
            self.font = self.font()
            self.font.setPointSize(40)
            self.setFont(self.font)

        parent.h2_layout.addWidget(self)

        self.resize(*widget_size)

        self.thread = self.FrameThread(self)
        self.thread.update_frame.connect(self.set_frame)
        self.thread.start()

    @pyqtSlot(QImage)
    def set_frame(self, image):
        self.setPixmap(QPixmap.fromImage(image))


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(None)
        self.title = '基于多尺度特征融合的高速公路违章行驶车辆识别与处理系统'
        self.location = [0, 0, 1280, 720]
        self.setting_list: SettingList[SettingItem] = SettingList()

        self._url = None
        self._conf_threshold = 0.5
        self._iou_threshold = 0.5

        self.icon = QIcon('icon.ico')
        self.setWindowIcon(self.icon)

        self.load_setting()
        self.init()

    @property
    def url(self):
        return self._url

    @property
    def conf_threshold(self):
        return self._conf_threshold

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @url.setter
    def url(self, url):
        self._url = url

        if hasattr(self, 'illeagal_list_widget'):
            self.illeagal_list_widget.deleteLater()
            del self.illeagal_list_widget

        if self.url is not None:
            self.illeagal_list_widget = QListWidget(self)
            self.illeagal_list_widget.setMaximumSize(300, 500)

            self.h2_layout.addWidget(self.illeagal_list_widget)

        # remove stream widget
        if hasattr(self, 'stream_widget'):
            if hasattr(self.stream_widget, 'detect_thread'):
                self.stream_widget.detect_thread.stop()
            if hasattr(self.stream_widget, 'dataset'):
                self.stream_widget.dataset.frame_thread.stop()
            self.stream_widget.deleteLater()
            del self.stream_widget

        self.stream_widget = StreamDetectorWidget(self)

    @conf_threshold.setter
    def conf_threshold(self, conf_threshold):
        self._conf_threshold = conf_threshold

    @iou_threshold.setter
    def iou_threshold(self, iou_threshold):
        self._iou_threshold = iou_threshold

    def load_setting(self, path: str = settings_path) -> None:
        if not pathlib.Path(path).exists():
            return

        with open(pathlib.Path(path), 'r', encoding='utf-8') as f:
            self.setting_list = SettingList(json.load(f, object_hook=lambda x: SettingItem(**x)))

    def save_setting(self, path: str = settings_path) -> None:
        with open(pathlib.Path(path), 'w', encoding='utf-8') as f:
            json.dump(self.setting_list, f, default=lambda x: x.__dict__(), ensure_ascii=False)

    def show_setting(self):
        self.setting_widget = SettingWidget(parent=self)
        self.setting_widget.exec()

    def init(self) -> None:
        self.setWindowTitle(self.title)
        self.setGeometry(*self.location)
        self.center(self)

        self.widget_setting_button = QPushButton('监控源设置', self)
        self.barrier_button = QPushButton('电子栅栏设置', self)
        self.threshold_button = QPushButton('阈值设置', self)
        self.quit_button = QPushButton('退出', self)

        self.widget_setting_button.setMaximumSize(100, 50)
        self.barrier_button.setMaximumSize(100, 50)
        self.threshold_button.setMaximumSize(100, 50)
        self.quit_button.setMaximumSize(100, 50)

        self.widget_setting_button.clicked.connect(self.show_setting)
        self.quit_button.clicked.connect(self.close)

        self.v_layout = QVBoxLayout(self)

        self.h_layout = QHBoxLayout()
        self.h2_layout = QHBoxLayout()

        self.h_layout.addWidget(self.widget_setting_button)
        self.h_layout.addWidget(self.barrier_button)
        self.h_layout.addWidget(self.threshold_button)
        self.h_layout.addWidget(self.quit_button)

        self.stream_widget = StreamWidget(self)


        self.v_layout.addLayout(self.h_layout)
        self.v_layout.addLayout(self.h2_layout)

        self.show()

    @staticmethod
    def center(self) -> None:
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    model = attempt_load('yolov7.pt', select_device('0'))
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())