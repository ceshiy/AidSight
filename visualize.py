import cv2
import os
import numpy as np

# COCO 80类名称
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 你的推理结果
result_dict = {
    'category_id': [6, 3, 3, 6, 6, 6, 3, 3, 3, 3, 3, 6, 6, 10, 6],
    'bbox': [
        [153.825, 29.407, 447.392, 178.647], [402.46, 556.013, 256.94, 311.75], [414.727, 264.155, 141.52, 124.843],
        [658.719, 29.757, 148.088, 199.877], [0.0, 711.038, 170.279, 293.754], [965.125, 2.059, 111.147, 125.682],
        [241.081, 961.05, 310.283, 111.309], [524.978, 304.692, 187.745, 154.036], [1135.591, 77.996, 86.025, 63.881],
        [0.267, 725.649, 166.926, 280.837], [1233.929, 137.068, 121.806, 120.896], [590.789, 35.189, 64.636, 105.936],
        [1130.98, 30.835, 105.021, 110.503], [1247.344, 52.698, 38.318, 65.836], [1136.332, 76.828, 89.904, 66.2]
    ],
    'score': [
        0.8986, 0.87972, 0.78466, 0.77596, 0.67827, 0.64301, 0.64187, 0.52055, 0.49287, 0.4663,
        0.43277, 0.33075, 0.32748, 0.2987, 0.28392
    ]
}

# 加载原始图像
image_path = "D:/MindYOLO/mindyolo/image/test1.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # 遍历每个检测结果
    for i in range(len(result_dict['category_id'])):
        x, y, w, h = result_dict['bbox'][i]
        score = result_dict['score'][i]
        cls_id = result_dict['category_id'][i]
        label = f"{COCO_NAMES[cls_id]} {score:.2f}"

        # 计算右下角坐标
        x2 = int(x + w)
        y2 = int(y + h)
        x1 = int(x)
        y1 = int(y)

        # 画矩形框（绿色）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画类别标签（白色文字，黑色背景）
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 保存结果
    output_path = "D:/MindYOLO/mindyolo/output/vis_test1.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"[INFO] Visualization saved to {output_path}")

    # 可选：显示图像（Windows 下会弹窗）
    # cv2.imshow("Detection Result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()