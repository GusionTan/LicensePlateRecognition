#coding:utf-8
import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel,QComboBox, QApplication,QFileDialog)
from PyQt5.QtGui import (QPainter, QPen, QFont, QPixmap)
from PyQt5.QtCore import Qt
import numpy
from Recognition import Run

class Tan(QWidget):
    #界面
    def __init__(self):
        super(Tan, self).__init__()

        self.resize(410, 380)
        self.move(100, 100)    
        self.setWindowTitle('基于深度学习的车牌识别系统的设计与实现')
        self.pic_path = ''

        self.setMouseTracking(False)

        self.btn_choice = QPushButton("选择图片", self)
        self.btn_choice.setGeometry(2, 12, 80, 35)
        self.btn_choice.clicked.connect(self.setBrowerPath)
        
        self.label_path = QLabel('', self)
        self.label_path.setGeometry(90, 12, 312, 35)
        self.label_path.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_path.setAlignment(Qt.AlignCenter)

        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 50, 400, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('结果：', self)
        self.label_result_name.setGeometry(2, 340, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 340, 120, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}") 
        self.label_result.setAlignment(Qt.AlignCenter) 

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(195, 340, 50, 35)
        self.btn_recognize.clicked.connect(self.reco)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(255, 340, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(315, 340, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    #点击识别，调用车牌识别函数
    def reco(self):
        tr = Run()
        str = tr.start(self.pic_path)
        self.label_result.setText(str)
        
    def btn_clear_on_clicked(self):
        self.label_path.setText('')
        self.label_result.setText('')
        self.label_draw.setPixmap(QPixmap(""))
        self.update()

    def btn_close_on_clicked(self):
        self.close()
    
    def setBrowerPath(self): 
        self.label_path.setText('')
        download_path = QFileDialog.getOpenFileName(self,"选择图片","E:\test_1")        
        self.pic_path = download_path[0]
        self.label_path.setText(self.pic_path)  
        pix = QPixmap(self.pic_path)
        self.label_draw.setPixmap(pix)
        self.label_draw.setScaledContents(True)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Tan()
    ex.show()
    app.exec_()