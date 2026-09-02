import numpy as np
import sys
import threading
import struct
import serial
import time
import matplotlib
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from serial.tools import list_ports
matplotlib.use("Qt5Agg")  # 使用 Qt5


reader1_mode_button = False
reader2_mode_button = False
reset_mode = False
target_rotation_matrix = None
target_lines = []
arduino_ports = []
reader_mode = False
is_recording = False
recorded_rotations = None
sensor_result = None


# threading 相關的部分
class SerialReader(threading.Thread):
    # 初始化threading並準備讀取sensor的值
    def __init__(self, port, arrays, filename, lock, arduino_ports):
        super().__init__()
        self.port = port
        self.arrays = arrays
        self.filename = filename
        self.data = np.eye(3)  # Initialize as identity matrix
        self.running = True
        self.data_send = False
        self.ser = serial.Serial(port, 115200, timeout=1)
        self.lock = lock
        self.arduino_ports = arduino_ports
        global reader1_mode_button
        global reader2_mode_button
        global reset_mode
        # global arduino_ports
        time.sleep(1)

    def run(self):
        global reader1_mode_button
        global reader2_mode_button
        global arduino_ports

        while self.running:
            while not self.data_send:
                self.send_data()

            while reader1_mode_button is True and self.port == self.arduino_ports[0]:
                self.send_command(1)
                self.receive_data()
            while reader2_mode_button is True and self.port == self.arduino_ports[1]:
                self.send_command(1)
                self.receive_data()
            while reset_mode:
                self.send_command(2)
                self.goto_reset()

            # 開啟校正資料，並傳遞給Arduino
            # 讀取sensor print出的值，並將其拆解再組合成3X3的矩陣
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                data = line.split(', ')
                if len(data) == 9:        # 這邊會檢測，讀到完整的9個數據的時候才會開始畫圖
                    try:
                        matrix = tuple(map(float, data))
                        matrix = np.array(matrix).reshape(3, 3)
                        with self.lock:
                            self.data = matrix
                    except ValueError as e:
                        print(f'Error converting data to float: {e}')

    def send_data(self):
        # 等待連接
        time.sleep(2)

        with open(self.filename, 'r') as f:
            for line in f:
                array = list(map(float, line.strip().split(', ')))
                self.arrays.append(array)

        for array in self.arrays:
            for number in array:
                self.ser.write(struct.pack('f', number))
            time.sleep(0.1)    # 給Arduino一些時間處裡數據

        self.data_send = True  # 告訴threading DATA已經傳送完畢

    def receive_data(self):
        global reader1_mode_button
        global reader2_mode_button
        global line

        # 顯示來自Arduino的提示字
        count = 0
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
                count += 1
                if count > 5:
                    break
            return line

        # 打開file並寫回資料
        with open(self.filename, 'w') as f:
            # 等待Arduino處裡數據並回傳
            while True:
                if self.ser.in_waiting >= 48:  # 檢查所有DATA都回來
                    data = self.ser.read(48)   # 讀取所有需要的DATA
                    for i in range(4):
                        returned_array = []
                        for j in range(3):
                            float_bytes = data[(i * 12) + (j * 4):(i * 12) + (j * 4) + 4]
                            number = struct.unpack('f', float_bytes)[0]
                            returned_array.append(number)
                        f.write(', '.join(f"{num:.2f}" for num in returned_array) + '\n')
                    break
                time.sleep(1)
            reader1_mode_button = False
            reader2_mode_button = False

    # 傳送指令給Arduino接收
    def send_command(self, mode):
        if mode == 1:
            self.ser.write('1'.encode())
        elif mode == 2:
            self.ser.write('22'.encode())

    # 將python變數改回正常模式
    def goto_reset(self):
        global reset_mode
        time.sleep(0.1)

        self.data_send = False
        reset_mode = False

    # 當程式停止時，threading 也要關閉
    def stop(self):
        self.running = False
        self.ser.close()


# PyQt6畫面使用的class
#class MainWindow(QMainWindow):
    # 初始化，設定一些基礎的參數，並且放上畫布以及按鈕
    #def __init__(self):
    #    super().__init__()

    #    self.setWindowTitle("3D Rotation Visualization")
    #    self.setGeometry(100, 100, 800, 600)

    #    central_widget = QWidget()
    #    self.setCentralWidget(central_widget)

    #    layout = QVBoxLayout(central_widget)

    #    self.figure = plt.figure()
    #    self.ax = self.figure.add_subplot(111, projection='3d')
    #    self.canvas = FigureCanvas(self.figure)
    #    layout.addWidget(self.canvas)

    #   self.reader1_mode_button = QPushButton("Set reader1 to Calibrate Mode")
    #    self.reader1_mode_button.clicked.connect(toggle_reader1_mode)
    #    layout.addWidget(self.reader1_mode_button)

    #    self.reader2_mode_button = QPushButton("Set reader2 to Calibrate Mode")
    #    self.reader2_mode_button.clicked.connect(toggle_reader2_mode)
    #    layout.addWidget(self.reader2_mode_button)

    #    self.reset_button = QPushButton("Reset Arduino")
    #    self.reset_button.clicked.connect(send_reset_command)
    #    layout.addWidget(self.reset_button)

    #    self.switch_button = QPushButton("Switch Port")
    #    self.switch_button.clicked.connect(switch_port)
    #    layout.addWidget(self.switch_button)

    #    self.record_button = QPushButton("Record Target Rotation")
    #    self.record_button.clicked.connect(self.record_target_rotation)
    #    layout.addWidget(self.record_button)

    #    self.quit_button = QPushButton("Quit")
    #    self.quit_button.clicked.connect(self.close)
    #    layout.addWidget(self.quit_button)

    #   self.rel_line_x, = self.ax.plot([0, 1], [0, 0], [0, 0], 'b', label='X-axis')
    #    self.rel_line_y, = self.ax.plot([0, 0], [0, 1], [0, 0], 'g', label='Y-axis')
    #    self.rel_line_z, = self.ax.plot([0, 0], [0, 0], [0, 1], 'r', label='Z-axis')
    #    self.rel_lines = [self.rel_line_x, self.rel_line_y, self.rel_line_z]

    #    self.ax.set_xlim(-1, 1)
    #    self.ax.set_ylim(-1, 1)
    #    self.ax.set_zlim(-1, 1)
    #    self.ax.set_xlabel('X')
    #    self.ax.set_ylabel('Y')
    #    self.ax.set_zlabel('Z')
    #    self.ax.legend()

    #    self.lock = threading.Lock()

        # 自動抓取arduino的ports
    #    global arduino_ports
    #    arduino_ports = find_arduino_ports()
    #    if len(arduino_ports) < 2:
    #        print("Arduino ports not found")
    #        sys.exit()

    #    port1 = arduino_ports[0]
    #    port2 = arduino_ports[1]

    #    self.array3 = []
    #    self.array5 = []

        # 開啟threading
    #    self.reader1 = SerialReader(port1, self.array3, 'calibrate_data_COM3.txt', self.lock)
    #    self.reader2 = SerialReader(port2, self.array5, 'calibrate_data_COM5.txt', self.lock)

    #    self.reader1.start()
    #    self.reader2.start()
        

        # 設定Timer讓他每40ms更新一次
    #    self.timer = QTimer(self)
    #    self.timer.timeout.connect(self.update_canvas)
    #    self.timer.start(40)  # 40毫秒更新一次

    # 更新圖畫
    def update_canvas(self):
        update_plot(self.ax, self.lock, self.reader1, self.reader2, self.rel_lines)
        self.canvas.draw()

    # 紀錄現在的旋轉量
    def record_target_rotation(self):
        global target_rotation_matrix
        global is_recording
        if not is_recording:  # 檢查是否已經在紀錄
            is_recording = True
            with self.lock:
                mat1 = self.reader1.data
                mat2 = self.reader2.data
                target_rotation_matrix = calculate_relative_rotation(mat1, mat2)
                self.draw_target_rotation(target_rotation_matrix)
            is_recording = False  # 記錄完後重置紀錄變數
        print("已記錄當前姿態")

    # 以虛線畫出紀錄的姿態
    def draw_target_rotation(self, rotation_matrix):
        global target_lines
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])

        target_x = rotation_matrix @ x
        target_y = rotation_matrix @ y
        target_z = rotation_matrix @ z

        if target_lines:
            for line in target_lines:
                line.remove()
            target_lines = []

        target_lines = [
            self.ax.plot([0, target_x[0]], [0, target_x[1]], [0, target_x[2]], 'b--')[0],
            self.ax.plot([0, target_y[0]], [0, target_y[1]], [0, target_y[2]], 'g--')[0],
            self.ax.plot([0, target_z[0]], [0, target_z[1]], [0, target_z[2]], 'r--')[0]
        ]
        # self.canvas.draw()

    # 當結束事件發生的時候要做的事
    def closeEvent(self, event):
        self.reader1.stop()
        self.reader2.stop()
        self.reader1.join()
        self.reader2.join()
        event.accept()

# 讓python半途現在是不是在reader1的校正模式
def toggle_reader1_mode():
    global reader1_mode_button
    reader1_mode_button = True

# 讓python判讀線在是不是在reader2的校正模式
def toggle_reader2_mode():
    global reader2_mode_button
    reader2_mode_button = True

# 讓python判別現在是不是在傳輸資料的狀態
def send_reset_command():
    global reset_mode
    reset_mode = True

# 看是port1做基準，還是port2做基準
def switch_port():
    global reader_mode
    reader_mode = not reader_mode


# 計算相對旋轉量，因為y, z兩軸是相反的，所以使用鏡射矩陣將負號補回去
def calculate_relative_rotation(mat1, mat2):
    global reader_mode
    match reader_mode:
        case False:
            rel = mat1 @ np.linalg.inv(mat2)
        case True:
            rel = mat2 @ np.linalg.inv(mat1)

    mirror_x = np.diag([-1, 1, 1])
    mirror_z = np.diag([1, 1, -1])
    rel = mirror_x @ rel @ mirror_x
    rel = mirror_z @ rel @ mirror_z

    return rel


# 更新畫布時的函式，會更新出新的三軸數據，讓畫布可以根據他來畫出新的三軸
def update_plot(ax, lock, reader1, reader2, rel_lines, dash_lines, true_point):
    global recorded_rotations
    global rel_rotation
    with lock:
        mat1 = reader1.data
        mat2 = reader2.data

    rel_rotation = calculate_relative_rotation(mat1, mat2)

    if recorded_rotations is not None:
        compare_to_target(rel_rotation, recorded_rotations)

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    x_rel, y_rel, z_rel = rel_rotation @ x, rel_rotation @ y, rel_rotation @ z

    for line, vec in zip(rel_lines, [x_rel, y_rel, z_rel]):
        line.set_data([0, vec[0]], [0, vec[1]])
        line.set_3d_properties([0, vec[2]])
        
    if true_point:
        recorded_rotations = calculate_relative_rotation(mat1, mat2)

        # new_lines = []
        # colors = ['b', 'g', 'r']
        # for i, vec in enumerate([x_rel, y_rel, z_rel]):
        #     line, = ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], colors[i], linestyle='--', linewidth=3, alpha=0.5)
        #     new_lines.append(line)
        # rel_lines.extend(new_lines)
        # 更新虛線並設置為可見
        for dash_line, vec in zip(dash_lines, [x_rel, y_rel, z_rel]):
            dash_line.set_data([0, vec[0]], [0, vec[1]])
            dash_line.set_3d_properties([0, vec[2]])
            dash_line.set_visible(True)

    return rel_rotation


# 跟目標矩陣做相對角的計算，並且轉回歐拉角分成三個方向的參數，取最大的偏移量，並且判讀要給使用者使用什麼引導語讓他回到紀錄的姿態
def compare_to_target(current_rotation, target_rotation):
    global reader_mode
    global sensor_result
    # 計算與目標矩陣相對的旋轉量
    match reader_mode:
        case False:
            relative_rotation = calculate_relative_rotation(current_rotation, target_rotation)
        case True:
            relative_rotation = calculate_relative_rotation(target_rotation, current_rotation)

    # 轉成歐拉角
    angles = extract_euler_angles(relative_rotation)

    # 根據角度給予調整建議
    pitch, roll, yaw = angles
    max_angel = max(abs(pitch), abs(roll), abs(yaw))
    if max_angel > 0.05:
        if max_angel == abs(pitch):
            if pitch > 0:
                sensor_result = str("前傾")
                # print("前傾")
            else:
                sensor_result = str("後傾")
                # print("後傾")
        elif max_angel == abs(roll):
            if roll > 0:
                sensor_result = str("後傾")
                # print("右傾")
            else:
                sensor_result = str("左傾")
                # print("左傾")
        elif max_angel == abs(yaw):
            if yaw > 0:
                sensor_result = str("逆時針旋轉")
                # print("逆時針旋轉")
            else:
                sensor_result = str("順時針旋轉")
                # print("順時針旋轉")
    else:
        sensor_result = str("correct direction")
        # print("correct direction")
        
    # 將結果寫入文件
    with open('sensor_result.txt', 'w', encoding='utf-8') as file:
        file.write(sensor_result)
    
    return sensor_result

# 將旋轉矩陣轉歐拉角
def extract_euler_angles(rotation_matrix):
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# 自動搜索arduino的ports
def find_arduino_ports(max_attempts=10, delay=1):
    for attempt in range(max_attempts):
        ports = list_ports.comports()
        arduino_port = []
        for port in ports:
            if 'Arduino' in port.description or 'CH340' in port.description:
                arduino_port.append(port.device)
        if len(arduino_port) >= 2:
            return arduino_port
        time.sleep(delay)
    return []


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    timer = QTimer()
    timer.timeout.connect(window.update_canvas)
    timer.start(40)  # 40 毫秒更新一次

    sys.exit(app.exec())