import cv2
import numpy as np
import os
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import copy

from models.experimental import attempt_load
from utils.datasets.yolo_datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def get_face_yolo(image, face_det_model, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img0 = copy.deepcopy(image)
    h0, w0 = image.shape[:2]
    r = 640 / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(640, s=face_det_model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = face_det_model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres=0.6, iou_thres=0.5)

    bboxes = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()

        for j in range(det.size()[0]):
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]), int(xyxy[3]) - int(xyxy[1]), conf]
            bboxes.append(bbox)

    if len(bboxes) == 0:
        return None

    bboxes.sort(key=lambda bbox: bbox[-1], reverse=True)
    return bboxes[0]


def show_results(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input


def parse_classify_model_name(model_name):
    info = model_name.split('.pth')[0].split('-')
    model_type = info[0]
    w_input, h_input = info[-1].split('_')
    return int(h_input), int(w_input), model_type


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_state_dict(model):
    return unwrap_model(model).state_dict()


def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w, cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def get_transforms(mode='train', label=0, img_width=240, img_height=240):
    if mode == 'train':
        transforms = [
            # real transform
            A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.05, 0.05), rotate_limit=5,
                                   border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.3),
                A.RandomScale(scale_limit=(0.9, 1.03), p=0.3),
                A.OneOf([
                    A.Blur(blur_limit=(3, 5), p=0.1),
                    A.GaussianBlur(p=0.1, blur_limit=(3, 5))
                ], p=0.2),
                A.ImageCompression(quality_lower=30, quality_upper=85, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.Resize(height=img_height, width=img_width, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], p=1.0),
            # recapture transform
            A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.05, 0.05), rotate_limit=5,
                                   border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.3),
                A.RandomScale(scale_limit=(0.9, 1.03), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.Resize(height=img_height, width=img_width, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], p=1.0)
        ]
        return transforms[label]
    else:
        transforms = A.Compose([
            A.Resize(height=img_height, width=img_width, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], p=1.0)
        return transforms


def get_recapture_transform(H, W):
    transforms = A.Compose([
        A.Resize(height=H, width=W, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transforms


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec
