import argparse
import ast
import math
import os
import sys
import time
import cv2
import numpy as np
import yaml
import urllib.parse
from datetime import datetime
import mindspore as ms
from mindspore import Tensor, nn
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh, process_mask_upsample, scale_image
from mindyolo.utils.utils import draw_result, set_seed

def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument(
        "--precision_mode", type=str, default=None, help="set accuracy mode of network model"
    )
    parser.add_argument("--weight", type=str, default="yolov7_300.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--exec_nms", type=ast.literal_eval, default=True, help="whether to execute NMS or not")
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_infer", help="save dir")
    parser.add_argument("--source", type=str, default="0", help="path to image, video file, or camera index (e.g., 0 for webcam)")
    parser.add_argument("--video_path", type=str, help="path to video or camera index (e.g., 0)")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    return parser

def set_default_infer(args):
    ms.set_context(mode=args.ms_mode)
    ms.set_recursion_limit(2000)
    if args.precision_mode is not None:
        ms.device_context.ascend.op_precision.precision_mode(args.precision_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.device_target == "Ascend":
        ms.set_device("Ascend", int(os.getenv("DEVICE_ID", 0)))
    args.rank, args.rank_size = 0, 1

    args.data.nc = 1 if args.single_cls else int(args.data.nc)
    args.data.names = ["item"] if args.single_cls and len(args.data.names) != 1 else args.data.names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )

    platform = sys.platform
    if platform == "win32":
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)

    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))

# ==============================
# 新增：在帧上绘制结果（返回带框图像）
# ==============================
def draw_result_on_frame(img, result_dict, class_names, is_coco_dataset=False, is_seg=False):
    from mindyolo.data import COCO80_TO_COCO91_CLASS
    frame = img.copy()
    h, w = frame.shape[:2]
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in class_names]

    for i in range(len(result_dict["category_id"])):
        cls_id_91 = result_dict["category_id"][i]
        bbox = result_dict["bbox"][i]
        score = result_dict["score"][i]

        # 关键：将 COCO91 ID 转回 0~79 的原始索引
        if is_coco_dataset:
            try:
                cls_idx = COCO80_TO_COCO91_CLASS.index(cls_id_91)
            except ValueError:
                continue  # 跳过无效类别
            label = f"{class_names[cls_idx]} {score:.2f}"
            color = colors[cls_idx]
        else:
            if cls_id_91 >= len(class_names):
                continue
            label = f"{class_names[cls_id_91]} {score:.2f}"
            color = colors[cls_id_91]

        x, y, w_box, h_box = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w_box), int(y + h_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if is_seg and "segmentation" in result_dict:
            mask = result_dict["segmentation"][i]
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(frame)
            overlay[:, :] = color
            masked_overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
            frame = cv2.addWeighted(frame, 1.0, masked_overlay, 0.5, 0)

    return frame

# ==============================
# detect / segment 函数保持不变（略，直接复用你提供的）
# ==============================

# （此处省略 detect 和 segment 函数，完全使用你提供的原始实现）

def detect(
    network: nn.Cell,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    exec_nms: bool = True,
    nms_time_limit: float = 60.0,
    img_size: int = 640,
    stride: int = 32,
    num_class: int = 80,
    is_coco_dataset: bool = True,
):
    # ...（完全复制你提供的 detect 函数内容）...
    # 为节省篇幅，此处不重复粘贴，实际使用时保留原函数
    h_ori, w_ori = img.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)
    _t = time.time()
    out, _ = network(imgs_tensor)
    out = out[-1] if isinstance(out, (tuple, list)) else out
    infer_times = time.time() - _t
    t = time.time()
    out = out.asnumpy()
    out = non_max_suppression(
        out,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
        need_nms=exec_nms,
    )
    nms_times = time.time() - t
    result_dict = {"category_id": [], "bbox": [], "score": []}
    total_category_ids, total_bboxes, total_scores = [], [], []
    for si, pred in enumerate(out):
        if len(pred) == 0:
            continue
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        category_ids, bboxes, scores = [], [], []
        for p, b in zip(pred.tolist(), box.tolist()):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))
        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)
    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)
    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")
    return result_dict

def segment(
    network: nn.Cell,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = 60.0,
    img_size: int = 640,
    stride: int = 32,
    num_class: int = 80,
    is_coco_dataset: bool = True,
):
    # ...（完全复制你提供的 segment 函数内容）...
    h_ori, w_ori = img.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)
    _t = time.time()
    out, (_, _, prototypes) = network(imgs_tensor)
    infer_times = time.time() - _t
    t = time.time()
    _c = num_class + 4 if conf_free else num_class + 5
    out = out.asnumpy()
    bboxes, mask_coefficient = out[:, :, :_c], out[:, :, _c:]
    out = non_max_suppression(
        bboxes,
        mask_coefficient,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
    )
    nms_times = time.time() - t
    prototypes = prototypes.asnumpy()
    result_dict = {"category_id": [], "bbox": [], "score": [], "segmentation": []}
    total_category_ids, total_bboxes, total_scores, total_seg = [], [], [], []
    for si, (pred, proto) in enumerate(zip(out, prototypes)):
        if len(pred) == 0:
            continue
        pred_masks = process_mask_upsample(proto, pred[:, 6:], pred[:, :4], shape=imgs_tensor[si].shape[1:])
        pred_masks = pred_masks.astype(np.float32)
        pred_masks = scale_image((pred_masks.transpose(1, 2, 0)), (h_ori, w_ori))
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        category_ids, bboxes, scores, segs = [], [], [], []
        for ii, (p, b) in enumerate(zip(pred.tolist(), box.tolist())):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))
            segs.append(pred_masks[:, :, ii])
        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)
        total_seg.extend(segs)
    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)
    result_dict["segmentation"].extend(total_seg)
    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)
    logger.info(f"Predict result is:")
    for k, v in result_dict.items():
        if k == "segmentation":
            logger.info(f"{k} shape: {v[0].shape}")
        else:
            logger.info(f"{k}: {v}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")
    return result_dict

# ==============================
# 主推理函数（支持图像和视频）
# ==============================
def is_url(path):
    """Check if the path is a valid URL (supports http/https/rtsp/rtmp)."""
    try:
        result = urllib.parse.urlparse(path)
        return all([result.scheme, result.netloc]) and result.scheme.lower() in ['http', 'https', 'rtsp', 'rtmp']
    except Exception:
        return False

def infer(args):
    set_seed(args.seed)
    set_default_infer(args)

    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    is_coco_dataset = "coco" in args.data.dataset_name

    # ================================
    # 判断输入源：图像 / 本地视频 / 摄像头 / 网络流
    # ================================
    source = args.source
    is_video = False
    cap = None
    img = None

    if source.isdigit():  # 本地摄像头索引（如 "0", "1"）
        cap = cv2.VideoCapture(int(source))
        is_video = True
    elif is_url(source):  # 网络流（RTSP/HTTP/HTTPS/RTMP）
        cap = cv2.VideoCapture(source)
        is_video = True
    elif os.path.isfile(source):  # 本地文件
        file_ext = source.lower()
        if file_ext.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mjpg', '.mjpeg', '.ts')):
            cap = cv2.VideoCapture(source)
            is_video = True
        else:
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Image not found or unsupported format: {source}")
            is_video = False
    else:
        raise ValueError(
            f"Invalid source: '{source}'. Supported sources:\n"
            "- Camera index (e.g., '0')\n"
            "- Local image (e.g., 'image.jpg')\n"
            "- Local video (e.g., 'video.mp4')\n"
            "- Network stream URL (e.g., 'rtsp://...', 'http://...')"
        )

    # ================================
    # 单图推理
    # ================================
    if not is_video:
        if args.task == "detect":
            result_dict = detect(
                network=network,
                img=img,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                conf_free=args.conf_free,
                exec_nms=args.exec_nms,
                nms_time_limit=args.nms_time_limit,
                img_size=args.img_size,
                stride=max(max(args.network.stride), 32),
                num_class=args.data.nc,
                is_coco_dataset=is_coco_dataset,
            )
            if args.save_result:
                draw_result(img, result_dict, args.data.names, args.save_dir, is_coco_dataset)
        elif args.task == "segment":
            result_dict = segment(
                network=network,
                img=img,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                conf_free=args.conf_free,
                nms_time_limit=args.nms_time_limit,
                img_size=args.img_size,
                stride=max(max(args.network.stride), 32),
                num_class=args.data.nc,
                is_coco_dataset=is_coco_dataset,
            )
            if args.save_result:
                draw_result(img, result_dict, args.data.names, args.save_dir, is_coco_dataset, is_seg=True)
        logger.info("Inference completed on image.")
        return

    # ================================
    # 视频/摄像头/网络流实时推理
    # ================================
    logger.info(f"Starting video inference from: {source}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame. End of stream or camera disconnected.")
            break

        # 推理
        if args.task == "detect":
            result_dict = detect(
                network=network,
                img=frame,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                conf_free=args.conf_free,
                exec_nms=args.exec_nms,
                nms_time_limit=args.nms_time_limit,
                img_size=args.img_size,
                stride=max(max(args.network.stride), 32),
                num_class=args.data.nc,
                is_coco_dataset=is_coco_dataset,
            )
            annotated_frame = draw_result_on_frame(frame, result_dict, args.data.names, is_coco_dataset)
        elif args.task == "segment":
            result_dict = segment(
                network=network,
                img=frame,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                conf_free=args.conf_free,
                nms_time_limit=args.nms_time_limit,
                img_size=args.img_size,
                stride=max(max(args.network.stride), 32),
                num_class=args.data.nc,
                is_coco_dataset=is_coco_dataset,
            )
            annotated_frame = draw_result_on_frame(frame, result_dict, args.data.names, is_coco_dataset, is_seg=True)
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # 显示结果
        cv2.imshow("MindYOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Video inference stopped.")

if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)