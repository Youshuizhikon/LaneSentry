import cv2
import numpy as np
import onnxruntime as ort
import time

COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie', 30: 'suitcase',
    31: 'frisbee', 32: 'skis', 33: 'snowboard', 34: 'sports ball', 35: 'kite',
    36: 'baseball bat', 37: 'baseball glove', 38: 'skateboard', 39: 'surfboard',
    40: 'tennis racket', 41: 'bottle', 42: 'wine glass', 43: 'cup', 44: 'fork',
    45: 'knife', 46: 'spoon', 47: 'bowl', 48: 'banana', 49: 'apple',
    50: 'sandwich', 51: 'orange', 52: 'broccoli', 53: 'carrot', 54: 'hot dog',
    55: 'pizza', 56: 'donut', 57: 'cake', 58: 'chair', 59: 'couch',
    60: 'potted plant', 61: 'bed', 62: 'dining table', 63: 'toilet', 64: 'tv',
    65: 'laptop', 66: 'mouse', 67: 'remote', 68: 'keyboard', 69: 'cell phone',
    70: 'microwave', 71: 'oven', 72: 'toaster', 73: 'sink', 74: 'refrigerator',
    75: 'book', 76: 'clock', 77: 'vase', 78: 'scissors', 79: 'teddy bear',
    80: 'hair drier', 81: 'toothbrush'
}

def img_preprocess(image) -> np.ndarray:
    """
    对直接获取的图片做推理需要的预处理
    :param image: 需要处理的ndarray图片
    :return: 处理完的ndarray (CHW)(1,C,H,W)
    """
    img = cv2.resize(image, (640, 640))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = img/255.0
    return img.astype(np.float32)

def yolo_body_preprocess(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
    """

    :param outputs: onnxruntime推理后的结果文件
    :param conf_thres: 置信度阈值
    :param iou_thres: opencviou阈值，用于检测重叠框
    :return: detections: np.array([x1,y1,x2,y2,conf,cls])
    """
    # 重塑和转置
    predictions = outputs.squeeze().T # (8400,84)

    # 获取数据
    boxes = predictions[:, :4]  # xywh
    scores = predictions[:, 4:] # 80类得分
    cls_ids = np.argmax(scores, axis=1) # 最高引索
    cls_scores = np.amax(scores, axis=1) # 最高置信度
    '''
    cv2的dnn模块自动过滤
    # 过滤低置信度
    mask = cls_scores > conf_thres
    boxs = boxes[mask]
    cls_ids = cls_ids[mask]
    cls_scores = cls_scores[mask]
    '''
    # cxcywh->xyxy
    boxes[:, 0] -= boxes[:, 2] / 2  # x1 = x_center - w/2
    boxes[:, 1] -= boxes[:, 3] / 2  # y1 = y_center - h/2
    boxes[:, 2] += boxes[:, 0]  # x2 = x1 + w
    boxes[:, 3] += boxes[:, 1]  # y2 = y1 + h

    # NMS过滤置信度并且排除冗余，输出位置引索序列
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=cls_scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )
    detections = np.zeros((len(boxes), 6))
    # 按照序列取出数据并打包至detections
    if len(indices)>0:
        detections = np.column_stack([boxes, cls_scores, cls_ids])[indices]
    return detections

def main():
    cap = cv2.VideoCapture(0)
    # 使用cpu初始化推理
    session = ort.InferenceSession('yolo11n.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    # 初始化计时器
    time_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print('摄像头载入失败')
            break
        frame = cv2.flip(frame, 1)
        # 预处理图像
        input_tensor = img_preprocess(frame)

        # 推理
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_tensor})[0]

        # 数据处理
        detections = yolo_body_preprocess(outputs, 0.5, 0.5)
        # 绘制
        for i in detections:
            x1, y1, x2, y2 = i[:4].astype(int)
            cls_id = i[5].astype(int)+1
            conf = i[4]
            name = COCO_CLASSES.get(cls_id, 'unknown')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{name}:{conf:.2f}', (x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)

        # 计算并显示fps
        time_count += 1
        fps = int(time_count / (time.time()-start_time))
        cv2.putText(frame, f"{fps=}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        cv2.imshow('det', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
