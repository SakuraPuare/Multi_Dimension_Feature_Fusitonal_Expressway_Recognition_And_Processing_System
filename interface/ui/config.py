# Form implementation generated from reading ui file '.\config.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(446, 184)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=Dialog)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 441, 181))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.SizeEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.SizeEdit.sizePolicy().hasHeightForWidth())
        self.SizeEdit.setSizePolicy(sizePolicy)
        self.SizeEdit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.SizeEdit.setDragEnabled(True)
        self.SizeEdit.setObjectName("SizeEdit")
        self.gridLayout.addWidget(self.SizeEdit, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.verticalLayoutWidget_2)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(parent=self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)
        self.label.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setIndent(0)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.IoUEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.IoUEdit.sizePolicy().hasHeightForWidth())
        self.IoUEdit.setSizePolicy(sizePolicy)
        self.IoUEdit.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.IoUEdit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.IoUEdit.setObjectName("IoUEdit")
        self.gridLayout.addWidget(self.IoUEdit, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=self.verticalLayoutWidget_2)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.ConfEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.ConfEdit.sizePolicy().hasHeightForWidth())
        self.ConfEdit.setSizePolicy(sizePolicy)
        self.ConfEdit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ConfEdit.setObjectName("ConfEdit")
        self.gridLayout.addWidget(self.ConfEdit, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.SaveButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.SaveButton.setObjectName("SaveButton")
        self.horizontalLayout.addWidget(self.SaveButton)
        self.CancelButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.CancelButton.setObjectName("CancelButton")
        self.horizontalLayout.addWidget(self.CancelButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.IoUEdit, self.ConfEdit)
        Dialog.setTabOrder(self.ConfEdit, self.SizeEdit)
        Dialog.setTabOrder(self.SizeEdit, self.SaveButton)
        Dialog.setTabOrder(self.SaveButton, self.CancelButton)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "阈值设置"))
        self.label_2.setText(_translate("Dialog", "Conf"))
        self.label.setText(_translate("Dialog", "IoU"))
        self.label_3.setText(_translate("Dialog", "图像大小"))
        self.SaveButton.setText(_translate("Dialog", "Save"))
        self.CancelButton.setText(_translate("Dialog", "Cancel"))