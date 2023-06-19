# coding=utf-8
import json
import sys
from typing import Union

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QTableWidgetItem, QLineEdit, QPushButton, QDialog, QHBoxLayout, QLabel

from ui import config as ui_config
from ui import main as ui_main
from ui import source as ui_source


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
        self.parent.size_threshold = float(self.SizeEdit.text())
        self.parent.save_to_config()
        self.dialog.close()


class SourceWindows(ui_source.Ui_dialog):
    def __init__(self, parent=None):
        self.parent = parent
        self.setting_list = SettingList()
        super().__init__()
        self.dialog = QtWidgets.QDialog(parent.widget)
        self.setupUi(self.dialog)

        self.AddSourceButton.clicked.connect(self.add_setting)
        self.EditSourceButton.clicked.connect(self.edit_setting)
        self.DelectSouceButton.clicked.connect(self.delete_setting)
        self.UpMoveButton.clicked.connect(self.up_move_setting)
        self.DownMoveButton.clicked.connect(self.down_move_setting)
        self.FlushSourceButton.clicked.connect(self.flush_setting)

        self.SaveButton.clicked.connect(self.save_setting)
        self.SelectButton.clicked.connect(self.select_setting)
        self.CancelButtoon.clicked.connect(self.dialog.close)

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
            types = types_edit.text()
            setting.name = name_edit.text()
            setting.url, _ = setting.load_url(url_edit.text(), types)

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
            layout.addWidget(QLabel('类型:', self.dialog))
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

    def save_setting(self):
        with open('settings.json', 'w', encoding='utf-8') as f:
            json.dump(self.setting_list, f, default=lambda x: x.__dict__(), ensure_ascii=False, indent=4)

    def load_setting(self):
        with open("settings.json", 'r', encoding='u8') as f:
            self.setting_list = SettingList(json.load(f, object_hook=lambda x: SettingItem(**x)))
        self.flush_setting()

    def flush_setting(self):
        self.tableWidget.setRowCount(len(self.setting_list))
        for i, setting in enumerate(self.setting_list):
            for j, key in enumerate(SettingItem.__keys__):
                item = QTableWidgetItem(str(getattr(setting, key)))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tableWidget.setItem(i, j, item)

    def save_config(self):
        with open('settings.json', 'w', encoding='utf-8') as f:
            json.dump(self.setting_list, f, default=lambda x: x.__dict__(), ensure_ascii=False, indent=4)

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

    def cancel_setting(self):
        self.parent.video_url = ''


class MainWindows(ui_main.Ui_Form):
    _video_url = ''
    _iou_threshold = 0.5
    _conf_threshold = 0.5
    _img_size = 640

    def __init__(self):
        super().__init__()
        self.widget = QtWidgets.QWidget()
        self.setupUi(self.widget)

        self.show_text('请设置视频源')
        self.init()

        self.video_url = ''

    @property
    def video_url(self):
        return self._video_url

    @video_url.setter
    def video_url(self, url):
        print(url)
        self._video_url = url
        if url == '':
            self.show_text('请设置视频源')
        else:
            self.show_text('视频源: {}'.format(url), 20)

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, value):
        self._iou_threshold = value

    @property
    def conf_threshold(self):
        return self._conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, value):
        self._conf_threshold = value

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, value):
        self._img_size = value

    def show_text(self, text, font_size=40):
        # set text
        self.label.setText(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        # font
        font = QFont()
        font.setPointSize(font_size)
        self.label.setFont(font)

    def init(self):
        self.sourceWindow = SourceWindows(self)
        self.settingWindow = ConfigWindow(self)
        # click to show source window
        self.SourceButton.clicked.connect(self.sourceWindow.dialog.show)
        self.thresholdButton.clicked.connect(self.settingWindow.dialog.show)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindows()
    ui.widget.show()
    sys.exit(app.exec())
