# coding=utf-8
import json
import pathlib
import sys
import time
from typing import Union, Tuple

import cv2
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QApplication, QMessageBox, QLineEdit, QDialog, \
    QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QAbstractItemView, QSizePolicy, QListWidget, QVBoxLayout

settings_path = pathlib.Path('settings.json')


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


class Detector:


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
            print(f'cap is opening `{self.url}`')
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


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(None)
        self.title = '基于多尺度特征融合的高速公路违章行驶车辆识别与处理系统'
        self.location = [0, 0, 1280, 720]
        self.setting_list: SettingList[SettingItem] = SettingList()

        self._url = None
        self._threshold = 0.5

        self.icon = QIcon('icon.ico')
        self.setWindowIcon(self.icon)

        self.load_setting()
        self.init()

    @property
    def url(self):
        return self._url

    @property
    def threshold(self):
        return self._threshold

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
            self.stream_widget.thread.stop()
            self.stream_widget.deleteLater()
            del self.stream_widget

        self.stream_widget = StreamWidget(self)

        # remove chioce widget
        # if hasattr(self, 'choice_widget'):

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

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
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
    # setting_list = SettingList([SettingItem(1, '测试', 'rtsp://'), SettingItem(2, '测试', 'rtsp://')])
    # print(max(setting_list))
    # with open('setting.json', 'w', encoding='utf-8') as f:
    #     json.dump(setting_list, f, ensure_ascii=False, indent=4,
    #               default=lambda x: x.__dict__() if isinstance(x, SettingItem) else x)

    pass
