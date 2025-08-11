import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk
from cnocr import CnOcr

from api.model_inference import model_inference


class LaneSentryApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        # 初始化变量
        self.cap = None
        self.ocr = None
        self.session_yolo = None
        self.session_plate = None
        self.btns: dict[str, tk.Button] = {}
        self.interval_time = 1  # 默认间隔时间1分钟
        # self.interval_time = 0.1  # 便于测试
        self.leftover_time = 60  # 剩余时间60秒
        # self.leftover_time = 5 # 便于测试

        self.create_widget()
        self.load_model()  # 加载模型

    def create_widget(self):
        """创建应用界面"""
        # 设置标题
        (tk.Label(self, text='LaneSentry', width=10, height=1, fg='black', font=("Helvetica", 12, "bold italic"))
         .pack())

        # 剩余时间计时显示
        self.time_label = tk.Label(self, text=f'检测剩余时间: {self.leftover_time}秒', font=("Helvetica", 10))
        self.time_label.pack(pady=2)

        # 按钮框架
        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=1)

        # 按钮设定
        btns_info = [('打开摄像头', '#F8F8FF', self.open_camera, 'open_camera', 8),
                     (f'修改间隔:{self.interval_time}min', '#98F5FF', self.config_interval_time, 'config_interval_time',
                      11),
                     ('退出', '#FF0000', self.on_exit, 'exit', 4)]
        for text, bg, command, name, width in btns_info:
            btn = tk.Button(
                button_frame,
                text=text,
                name=name,
                bg=bg, fg='black',
                command=command,
                relief=tk.RAISED, bd=2,
                width=width,
                height=1,
            )
            btn.pack(padx=10, side=tk.LEFT)
            self.btns[name] = btn

        # 摄像头显示区域
        video_frame = tk.Frame(self, bg="#f0f0f0").pack(pady=5)
        self.camera_label = tk.Label(video_frame, name='camera_img', width=240, height=240)
        self.camera_label.pack()

    def load_model(self):
        """加载模型"""
        self.ocr = CnOcr()
        self.session_yolo = ort.InferenceSession('./det_model/carbusvansother.onnx',
                                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.session_plate = ort.InferenceSession('./det_model/car_plate_det.onnx',
                                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def open_camera(self):
        """打开摄像头"""
        self.btns['open_camera'].config(text='关闭摄像头', command=self.close_camera)  # 修改按钮文本为关闭
        try:
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            tk.messagebox.showerror("错误", f"无法打开摄像头: {e}")
        # 在摄像头打开后开始计时
        self.reduce_leftover_time()

        self.updata_video()

    def close_camera(self):
        """关闭摄像头"""
        self.btns['open_camera'].config(text='打开摄像头', command=self.open_camera)
        self.release_camera()
        self.camera_label.config(image='')  # 清空摄像头显示区域

    def updata_video(self):
        """更新摄像头画面"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(cv2.resize(frame, (240, 240)), cv2.COLOR_BGR2RGB)[::, ::-1, ::]  # 转置
                # 将图像转换为PIL格式
                img = Image.fromarray(frame_rgb)
                # 转换为Tkinter可用的格式
                img_tk = ImageTk.PhotoImage(img)
                # 更新标签显示
                self.camera_label.config(image=img_tk)
                self.camera_label.image = img_tk
                self.master.after(10, self.updata_video)  # 每10毫秒更新一次
            else:
                tk.messagebox.showerror("错误", "无法读取摄像头画面")
        else:
            tk.messagebox.showinfo("信息", "摄像头已关闭")

    def config_interval_time(self):
        """修改间隔时间"""
        self.interval_time += 2
        # 最大间隔时间10分钟
        if self.interval_time > 10:
            self.interval_time = 1
        # 更新按钮文本
        self.leftover_time = self.interval_time * 60  # 更新剩余时间
        self.btns['config_interval_time'].config(text=f'修改间隔:{self.interval_time}min')

    def reduce_leftover_time(self):
        """减少剩余时间"""
        # 检测摄像头是否开启
        if self.cap is None or not self.cap.isOpened():
            return
        # 更新按钮文本
        self.btns['config_interval_time'].config(text=f'修改间隔:{self.interval_time}min')

        self.leftover_time -= 1
        self.time_label.config(text=f'检测剩余时间: {self.leftover_time}秒')
        # 开启检测函数
        self.check_leftover_time()
        # 每秒减少一次
        self.master.after(1000, self.reduce_leftover_time)

    def check_leftover_time(self):
        """检查剩余时间是否为0"""
        if self.leftover_time <= 0:
            self.leftover_time = int(self.interval_time * 60)  # 重置剩余时间
            tk.messagebox.showinfo("信息", "检测时间已到，开始模型推理")
            # 获取当前摄像头画面并进行模型推理
            self.take_photo_and_identify()
        else:
            self.master.after(1000, self.check_leftover_time)

    def take_photo_and_identify(self):
        frame = None
        if self.cap is not None and self.cap.isOpened():
            frame = np.array(ImageTk.getimage(self.camera_label.image))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
            frame = cv2.imread('./test/car.jpg')  # 便于测试
            if frame is not None:
                self.run_model_inference(frame)
            else:
                tk.messagebox.showerror("错误", "无法读取摄像头画面")

    def run_model_inference(self, frame: np.ndarray):
        result = None
        try:
            result = model_inference(frame, self.session_yolo, self.session_plate, self.ocr)
        except Exception as e:
            tk.messagebox.showerror("错误", f"模型推理失败: {e}")
        # cv2.imshow('识别结果', result[0])  # test: 显示识别结果
        # cv2.waitKey(0)
        if result[1] is not None:
            # 处理车辆数据
            for car_id, pconf in result[1].items():
                tk.messagebox.showinfo('识别到车辆！', f"车辆ID: {car_id}, 置信度: {pconf:.2f}")
                print(f"车辆ID: {car_id}, 置信度: {pconf:.2f}")
        if result[2] is not None:
            # 处理车牌数据
            for car_id, (plate_text, pconf) in result[2].items():
                tk.messagebox.showinfo('识别到车牌！', f"车牌ID: {car_id}, 车牌文本: {plate_text}, 置信度: {pconf:.2f}")
                print(f"车牌ID: {car_id}, 车牌文本: {plate_text}, 置信度: {pconf:.2f}")
        else:
            print('未识别到车辆')

    def release_camera(self):
        """释放摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_exit(self):
        """退出程序"""
        self.release_camera()
        root.destroy()


root = tk.Tk()
root.title('LaneSentry')
root.geometry('240x400')

app = LaneSentryApp(root)

root.mainloop()
