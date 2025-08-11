import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import ImageDraw, Image, ImageFont
from cnocr import CnOcr

from api.EmailSend import send_email


def cxcywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    将cxcywh格式的框转换为xyxy格式
    :param boxes: 输入的框，格式为(cx, cy, w, h)
    :return: 转换后的框，格式为(x1, y1, x2, y2)
    """
    boxes[:, 0] -= boxes[:, 2] / 2  # x1 = x_center - w/2
    boxes[:, 1] -= boxes[:, 3] / 2  # y1 = y_center - h/2
    boxes[:, 2] += boxes[:, 0]  # x2 = x1 + w
    boxes[:, 3] += boxes[:, 1]  # y2 = y1 + h
    return boxes


def img_preprocess(image) -> np.ndarray:
    """
    对直接获取的图片做推理需要的预处理
    :param image: 需要处理的ndarray图片
    :return: 处理完的ndarray (CHW)(1,C,H,W)
    """
    img = cv2.resize(image, (640, 640))  # 始终调整为640x640
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = img / 255.0
    return img.astype(np.float32)


def yolo_body_preprocess(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
    """

    :param outputs: onnxruntime推理后的结果文件
    :param conf_thres: 置信度阈值
    :param iou_thres: opencviou阈值，用于检测重叠框
    :return: detections: np.ndarray([x1,y1,x2,y2,conf,cls])
    """
    # 重塑和转置
    predictions = outputs.squeeze().T  # (8400,8)

    # 获取数据
    boxes = predictions[:, :4]  # xywh
    scores = predictions[:, 4:]  # 4类得分
    cls_ids = np.argmax(scores, axis=1)  # 最高引索
    cls_scores = np.amax(scores, axis=1)  # 最高置信度
    # cxcywh->xyxy
    boxes = cxcywh2xyxy(boxes)

    # NMS过滤置信度并且排除冗余，输出位置引索序列
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=cls_scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )
    detections = np.zeros((len(boxes), 6))
    # 按照序列取出数据并打包至detections
    if len(indices) > 0:
        detections = np.column_stack([boxes, cls_scores, cls_ids])[indices]
        # 只获取前三项 car, bus, vans
        detections = detections[detections[:, 5] < 3]
    return detections


def yolo_plate_preprocess(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
    """
    车牌检测数据处理
    :param outputs: onnxruntime推理后车牌的结果文件
    :param conf_thres: 置信度阈值
    :param iou_thres: opencviou阈值，用于检测重叠框
    :return: detections: np.array([x1,y1,x2,y2,conf])
    """
    # 重塑和转置
    predictions = outputs.squeeze().T  # (8400,5)

    # 获取数据
    boxes = predictions[:, :4]  # xywh
    scores = predictions[:, 4]  # 得分

    # cxcywh->xyxy
    boxes = cxcywh2xyxy(boxes)

    # NMS过滤置信度并且排除冗余，输出位置引索序列
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )

    detections = np.zeros((5,))  # 默认返回空值
    if len(indices) > 0:
        # [0]保证拿到最高得分的数据
        detections = np.column_stack([boxes, scores])[indices][0]
        print(f'预处理信息（x1,y1,x2,y2,conf）：{detections}')
    return detections


def model_inference(img: np.ndarray, session_yolo: ort.InferenceSession, session_plate: ort.InferenceSession,
                    ocr: CnOcr) -> tuple:
    """
    对图片中的车辆和车牌进行检测并识别
    :param img_path: 图片文件数组
    :param session_yolo: 车辆检测的onnxruntime会话
    :param session_plate: 车牌检测的onnxruntime会话
    :param ocr: CnOcr实例，用于车牌文字识别
    :return: tuple(检测后的图片: ndarray, 车辆信息字典: dict, 车牌信息字典: dict)，
    车辆字典格式为{车辆ID: 置信度: float}，
    车牌字典格式为{车辆ID: (车牌号码: str, 置信度: float)}
    """
    start_time = time.time()

    # 初始化汽车检测yolo11n
    input_name_yolo = session_yolo.get_inputs()[0].name
    output_name_yolo = session_yolo.get_outputs()[0].name
    # 初始化车牌检测
    input_name_plate = session_plate.get_inputs()[0].name
    output_name_plate = session_plate.get_outputs()[0].name

    # 预处理图像
    input_tensor = img_preprocess(img)
    # 推理
    outputs = session_yolo.run([output_name_yolo], {input_name_yolo: input_tensor})[0]

    # 数据处理
    cars = yolo_body_preprocess(outputs, 0.5, 0.5)

    # 检测数据初始化
    car_id = 0
    plate_num = 0  # 检测到的车牌数量
    plate_data = {}
    car_data = {}
    # 绘制
    for car in cars:
        x1, y1, x2, y2 = car[:4].astype(int)
        conf = car[4]
        if conf == 0:  # 没有车辆时
            continue

        # 计算原始图像和处理图像的缩放比例
        img_height, img_width = img.shape[:2]
        scale_x = img_width / 640
        scale_y = img_height / 640

        # 将坐标缩放回原始图像尺寸
        draw_x1 = int(x1 * scale_x)
        draw_y1 = int(y1 * scale_y)
        draw_x2 = int(x2 * scale_x)
        draw_y2 = int(y2 * scale_y)
        # 绘制车辆区域
        cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
        cv2.putText(img, f'id:{car_id}:{conf:.2f}', (draw_x1, draw_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 3)
        car_data[car_id] = conf

        '''车牌检测'''
        car_region = img[draw_y1:draw_y2, draw_x1:draw_x2]
        # 检查裁剪区域是否为空
        if car_region.size == 0:
            car_id += 1
            # print("车牌区域为空，跳过处理")
            continue
        input_plate_tensor = img_preprocess(car_region)
        # 推理
        plate_outputs = session_plate.run([output_name_plate], {input_name_plate: input_plate_tensor})[0]
        # 车牌数据预处理
        plate = yolo_plate_preprocess(plate_outputs, 0.5, 0.5)

        '''车牌信息处理'''
        pconf = plate[4]
        if pconf == 0:  # 当车牌不存在时
            car_id += 1
            continue

        px1, py1, px2, py2 = plate[:4].astype(int)
        # 计算车牌在原始图像中的绝对坐标
        abs_px1 = draw_x1 + int(px1 * (draw_x2 - draw_x1) / 640)
        abs_py1 = draw_y1 + int(py1 * (draw_y2 - draw_y1) / 640)
        abs_px2 = draw_x1 + int(px2 * (draw_x2 - draw_x1) / 640)
        abs_py2 = draw_y1 + int(py2 * (draw_y2 - draw_y1) / 640)

        # 文字检测
        plate_region = img[abs_py1:abs_py2, abs_px1:abs_px2][:, :, ::-1]  # BGR转RGB
        plate_text = ocr.ocr(plate_region)[0]['text']
        # 绘制文字
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt_img = Image.fromarray(img_rgb)
        font = ImageFont.truetype("../sundries/simsun.ttc", 20)
        draw = ImageDraw.Draw(plt_img)
        draw.text((abs_px1, abs_py1 - 20), plate_text, font=font, fill=(255, 0, 0))

        img = cv2.cvtColor(np.array(plt_img), cv2.COLOR_RGB2BGR)
        # 绘制车牌框
        cv2.rectangle(img, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 0, 0), 2)
        # 向数组添加信息
        plate_data[car_id] = (plate_text, pconf.astype(float))
        # 发送邮件
        send_email(True, img, plate_text, pconf)
        # 增加识别数量
        car_id += 1
        plate_num += 1

    print(f'运行时间：{(time.time() - start_time):.2f}')
    print(f'返回数据dict(id:tuple(plate_text: str, pconf: float)\n:{plate_data}')
    # cv2.imshow('det', img)
    # cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 返回数据
    return (img, car_data, plate_data)


if __name__ == '__main__':
    img_array = cv2.imread('../test/car4.jpg')
    ocr = CnOcr()
    session_yolo = ort.InferenceSession('../det_model/carbusvansother.onnx',
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    session_plate = ort.InferenceSession('../det_model/car_plate_det.onnx',
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    resp = model_inference(img_array, session_yolo, session_plate, ocr)
    cv2.imshow('result', resp[0])
    print(resp[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
