from PyQt6 import QtWidgets, QtGui, QtCore  #建立圖形使用者介面 (GUI)
from PyQt6.QtWidgets import QSizePolicy
import sys  #與 Python 的系統參數和功能交互
import ctypes  #載入和調用 DLL（動態連結庫）
from ctypes import *
import warnings #管理警告訊息
import threading
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from setGUI import GUI

app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet("QWidget { color: #111111; }")
warnings.filterwarnings('ignore') #忽略所有警告訊息

# Define thread function return value
class DLLThread(threading.Thread):  
    def __init__(self, func, args=()): #__init_構造函數自動初始化屬性，不需要手動設置
        super(DLLThread, self).__init__() #super() 用於在子類中調用父類的方法
        self.func = func
        self.args = args
    def run(self):  #定義執行緒的執行內容，執行傳入的函數並存儲返回值
        self.result = self.func(*self.args)
    def get_result(self):  # 獲取函數執行的結果
        threading.Thread.join(self) #確保執行緒執行完後，主程式才繼續執行
        try:
            return self.result #成功執行時，返回存儲在 self.result 中的結果
        except:
            return None

class LoadDLL: #初始化載入 DLL 的類別，並初始化w和h表示寬與高
    def __init__(self):
        self.w = 600
        self.h = 600

class MainWindow(QtWidgets.QMainWindow): #應用程式的主窗口
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("小兒脊椎超音波影像即時追蹤與標註系統")
        screen =  QtWidgets.QApplication.screens() #獲取螢幕資訊，用於動態調整窗口大小
        screen_size = screen[0].size() #主螢幕的大小
        width = screen_size.width() - 40
        self.resize(width, 800)
        
        self.load_label = QtWidgets.QLabel(self)
        label_style = """
            QLabel {
                color: black;   
            }
        """ 
        self.load_label.setGeometry(375, 470, 650, 40) #位置: 左上角的 x=375 和 y=470。大小: 寬度=650，高度=40
        self.load_label.setText('請等待系統初始化...')
        self.load_label.setStyleSheet(label_style)
        self.load_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter) #將文字對齊到標籤的中央

        font = QtGui.QFont()
        font.setFamily('Microsoft JhengHei UI')
        font.setPointSize(32)
        font.setBold(True)
        self.load_label.setFont(font)
        
        self.bar = QtWidgets.QProgressBar(self)
        self.bar.setGeometry(500, 350, 400, 30)
        self.bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bar.setRange(0, 100) #創建進度條元件，用於顯示載入進度並設定進度條的值範圍為 0 到 100
        self.bar.setFormat('')
        self.bar.setStyleSheet('''
            QProgressBar {
                border: 2px solid #054C78;
                text-align:center;
                background:#B7FFD8;
                color:#FFFFFF;
                height: 30px; 
                border-radius: 8px;
                width:300px;
            }
            QProgressBar::chunk {
                background: #054C78;
                width:1px;
            }
        ''')
        
        self.load = 1
        self.load_label.hide() #隱藏初始化文字標籤
        self.bar.hide() #隱藏進度條
        self.show()
        
    def loading(self):
        value = 20
        while self.load == 1:
            k = random.randint(5, 10) #從 5 到 10 之間生成一個隨機整數，表示本次遞增的進度值
            value = value + k
            self.bar.setValue(value)
            time.sleep(1)
            if self.load == 0:
                break
        
        self.load_label.hide()
        self.bar.hide()
        
    def closeEvent(self, event):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("退出系統")
        msg.setText("您是否要退出系統?")
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)

        # 方法一：用樣式表（最直觀）
        msg.setStyleSheet("""
            QMessageBox { background-color: #FFFFFF; color: #111111; }
            QMessageBox QLabel { color: #111111; font-size: 14px; }
            QMessageBox QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1px solid #111111;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QMessageBox QPushButton:hover { background-color: #F5F5F5; }
            QMessageBox QPushButton:pressed { background-color: #EAEAEA; }
        """)

        reply = msg.exec()

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


load_dll = LoadDLL()  #創建一個 LoadDLL 類的實例，並賦值給變數 load_dll
root = MainWindow()
root.setStyleSheet("background-color: white")

run = GUI(app, root)

sys.exit(app.exec())
