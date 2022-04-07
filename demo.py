import argparse
import sys
from time import time

import cv2
import numpy as np
import torch
from torch.backends import cudnn
from torchvision import transforms

from utils import draw

sys.path.append("./hnet")
from hnet.backbone import HybridNetsBackbone
from hnet.utils.plot import (
    STANDARD_COLORS,
    get_index_label,
    plot_one_box,
    standard_to_bgr,
)
from hnet.utils.utils import (
    BBoxTransform,
    ClipBoxes,
    Params,
    boolean_string,
    letterbox,
    postprocess,
    restricted_float,
    scale_coords,
)

parser = argparse.ArgumentParser("HybridNets: End-to-End Perception Network - DatVu")
parser.add_argument("-p", "--project", type=str, default="bdd100k", help="Project file that contains parameters")
parser.add_argument("-c", "--compound_coef", type=int, default=3, help="Coefficient of efficientnet backbone")
parser.add_argument("--source", type=str, default="hnet/demo/video/1.mp4", help="The demo video file")
# parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument("-w", "--load_weights", type=str, default="hnet/weights/hybridnets.pth")
parser.add_argument("--nms_thresh", type=restricted_float, default="0.25")
parser.add_argument("--iou_thresh", type=restricted_float, default="0.3")
parser.add_argument("--cuda", type=boolean_string, default=True)
parser.add_argument("--float16", type=boolean_string, default=True, help="Use float16 for faster inference")

args = parser.parse_args()
compound_coef = args.compound_coef
source = args.source
# output = args.output
weight = args.load_weights
threshold = args.nms_thresh
iou_threshold = args.iou_thresh
use_cuda = args.cuda
use_float16 = args.float16

params = Params(f"hnet/projects/{args.project}.yml")
anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales
obj_list = params.obj_list
seg_list = params.seg_list
resized_shape = params.model["image_size"]

cudnn.fastest = True
cudnn.benchmark = True

color_list = standard_to_bgr(STANDARD_COLORS)
if isinstance(resized_shape, list):
    resized_shape = max(resized_shape)
normalize = transforms.Normalize(mean=params.mean, std=params.std)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)

# Load Model
model = HybridNetsBackbone(
    compound_coef=compound_coef,
    num_classes=len(obj_list),
    ratios=eval(anchors_ratios),
    scales=eval(anchors_scales),
    seg_classes=len(seg_list),
)
try:
    model.load_state_dict(
        torch.load(weight, map_location="cuda" if use_cuda else "cpu")
    )
except:
    model.load_state_dict(
        torch.load(weight, map_location="cuda" if use_cuda else "cpu")["model"]
    )

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# Set capture source
cap = cv2.VideoCapture(source)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h0, w0 = frame.shape[:2]  # orig hw
    r = resized_shape / max(h0, w0)  # resize image to img_size
    input_img = cv2.resize(
        frame, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA
    )
    h, w = input_img.shape[:2]

    (input_img, _, _), ratio, pad = letterbox(
        (input_img, input_img.copy(), input_img.copy()),
        resized_shape,
        auto=True,
        scaleup=False,
    )

    shapes = ((h0, w0), ((h / h0, w / w0), pad))

    if use_cuda:
        x = transform(input_img).cuda()
    else:
        x = transform(input_img)

    x = x.to(torch.float32 if not use_float16 else torch.float16)
    x.unsqueeze_(0)

    with torch.no_grad():
        features, regression, classification, anchors, seg = model(x)

        seg = seg[:, :, 12:372, :]
        da_seg_mask = torch.nn.functional.interpolate(
            seg, size=[h0, w0], mode="nearest"
        )
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask_ = da_seg_mask[0].squeeze().cpu().numpy().round()

        # # Lane Mask
        mask_lane = np.zeros_like(da_seg_mask_, dtype=np.uint8)
        # Choose faster method
        # mask_lane[np.unravel_index(np.where(da_seg_mask_.ravel() == 2), da_seg_mask_.shape)] = 255
        mask_lane[da_seg_mask_ == 2] = 255
        # cv2.imshow('lane', mask_lane)

        # # Drivable Segment Mask
        mask_drivable = np.zeros_like(da_seg_mask_, dtype=np.uint8)
        # Choose faster method
        # mask_drivable[np.unravel_index(np.where(da_seg_mask_.ravel() == 1), da_seg_mask_.shape)] = 255
        mask_drivable[da_seg_mask_ == 1] = 255
        # cv2.imshow('drivable', mask_drivable)

        # Combine Masks

        ## Fill Mask Color
        bgr_mask_lane = cv2.cvtColor(mask_lane, cv2.COLOR_GRAY2BGR)
        mask_bg_lane = np.full_like(frame, (255, 255, 0))
        bgr_mask_lane = cv2.bitwise_and(bgr_mask_lane, mask_bg_lane)

        bgr_mask_drivable = cv2.cvtColor(mask_drivable, cv2.COLOR_GRAY2BGR)
        mask_bg_drivable = np.full_like(frame, (255, 0, 255))
        bgr_mask_drivable = cv2.bitwise_and(bgr_mask_drivable, mask_bg_drivable)

        ## Combine Colored Masks
        frame = cv2.bitwise_and(frame, cv2.bitwise_not(bgr_mask_lane))
        frame = cv2.bitwise_and(frame, cv2.bitwise_not(bgr_mask_drivable))

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(
            x,
            anchors,
            regression,
            classification,
            regressBoxes,
            clipBoxes,
            threshold,
            iou_threshold,
        )
        out = out[0]
        out["rois"] = scale_coords(frame[:2], out["rois"], shapes[0], shapes[1])
        for j in range(len(out["rois"])):
            x1, y1, x2, y2 = out["rois"][j].astype(int)
            obj = obj_list[out["class_ids"][j]]
            score = float(out["scores"][j])
            plot_one_box(
                frame,
                [x1, y1, x2, y2],
                label=obj,
                score=score,
                color=color_list[get_index_label(obj, obj_list)],
            )

        frame, fps = draw.fpsmeter(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

cap.release()
