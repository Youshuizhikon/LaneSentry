import time

import cv2
import numpy as np
import onnxruntime as ort


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
    :return: detections: np.array([x1,y1,x2,y2,conf,cls])
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

    detections = np.zeros((len(boxes), 5))
    if len(indices) > 0:
        detections = np.column_stack([boxes, scores])[indices]
    return detections


def main(img_path: str):
    start_time = time.time()

    img = cv2.imread(img_path)
    # 使用cuda,cpu初始化推理
    # 初始化汽车检测yolo11n
    session_yolo = ort.InferenceSession('carbusvansother.onnx',
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name_yolo = session_yolo.get_inputs()[0].name
    output_name_yolo = session_yolo.get_outputs()[0].name
    # 初始化车牌检测
    session_plate = ort.InferenceSession('car_plate_det.onnx',
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name_plate = session_plate.get_inputs()[0].name
    output_name_plate = session_plate.get_outputs()[0].name

    # 预处理图像
    input_tensor = img_preprocess(img)
    # 推理
    outputs = session_yolo.run([output_name_yolo], {input_name_yolo: input_tensor})[0]

    # 数据处理
    detections = yolo_body_preprocess(outputs, 0.5, 0.5)
    # 绘制
    for car in detections:
        x1, y1, x2, y2 = car[:4].astype(int)
        cls_id = car[5].astype(int)
        name = 'car'  # 获取类别名称
        conf = car[4]

        # 计算原始图像和处理图像的缩放比例
        img_height, img_width = img.shape[:2]
        scale_x = img_width / 640
        scale_y = img_height / 640

        # 将坐标缩放回原始图像尺寸
        draw_x1 = int(x1 * scale_x)
        draw_y1 = int(y1 * scale_y)
        draw_x2 = int(x2 * scale_x)
        draw_y2 = int(y2 * scale_y)

        cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
        cv2.putText(img, f'{name}:{conf:.2f}', (draw_x1, draw_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 3)


        # 车牌检测
        car_region = img[draw_y1:draw_y2, draw_x1:draw_x2]
        # 检查裁剪区域是否为空
        if car_region.size == 0:
            # print("车牌区域为空，跳过处理")
            continue
        input_plate_tensor = img_preprocess(car_region)
        # 推理
        plate_outputs = session_plate.run([output_name_plate], {input_name_plate: input_plate_tensor})[0]
        # 车牌数据处理
        plates = yolo_plate_preprocess(plate_outputs, 0.5, 0.5)
        for plate in plates:
            px1, py1, px2, py2 = plate[:4].astype(int)
            pconf = plate[4]

            # 计算车牌在原始图像中的绝对坐标
            abs_px1 = draw_x1 + int(px1 * (draw_x2 - draw_x1) / 640)
            abs_py1 = draw_y1 + int(py1 * (draw_y2 - draw_y1) / 640)
            abs_px2 = draw_x1 + int(px2 * (draw_x2 - draw_x1) / 640)
            abs_py2 = draw_y1 + int(py2 * (draw_y2 - draw_y1) / 640)
            if pconf != 0:
                # 绘制车牌框
                cv2.rectangle(img, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 0, 0), 2)
                cv2.putText(img, f'plate:{pconf:.2f}', (abs_px1, abs_py1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)

    print(time.time()-start_time)
    cv2.imshow('det', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('car_label.png')
