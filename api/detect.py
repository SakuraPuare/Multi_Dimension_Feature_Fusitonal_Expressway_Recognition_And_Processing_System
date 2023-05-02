import random

import cv2
import torch
from torch.backends import cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_imshow, non_max_suppression, check_img_size, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def default(value, default_value, **kwargs):
    return default_value if kwargs.get(value, None) is not None else value


class Detector:
    def __init__(self) -> None:

        self.can_imshow = True
        self.agnostic_nms = False
        self.classes = None
        self.iou_threshold = 0.45
        self.conf_threshold = 0.25
        self.augment = False
        self.trace = True
        self.img_size = 640
        self.weights = 'yolov7_raw.pt'
        self.view_img = True
        self.device = select_device('0')

        self.load()

    def load(self) -> None:
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # load FP32 model
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # if self.trace:
        #     self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            self.model.half()

        pass

    @staticmethod
    def check_source_webcam(source: str) -> bool:
        if source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://')):
            return True
        return False

    def detect(self, source: str) -> None:
        self.webcam = self.check_source_webcam(source=source)
        if self.webcam:
            self.can_imshow = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=self.img_size, stride=self.stride)
        else:
            self.dataset = LoadImages(source, img_size=self.img_size, stride=self.stride)

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # if self.device.type != 'cpu':
        #     self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
        #         next(self.model.parameters())))  # run once
        # old_img_w = old_img_h = self.img_size
        # old_img_b = 1

        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]

            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=self.classes,
                                       agnostic=self.agnostic_nms)

            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                # print(f'{s}Done. ')

                # Stream results
                if self.can_imshow:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond


if __name__ == '__main__':
    detector = Detector()
    # detector.detect(r'D:\algorithm\dataset\raw\images\1595034896226.jpg')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8194/live/HD_a66b53d5-ee24-4f40-8151-e761ca39a6b6.m3u8?auth_key=1682950512-0-0-a77bc29bf7d530472c1b4cfbcbbaf9ef')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8097/live/HD_0232adab-423a-f287-5212-7896060397c4.m3u8?auth_key=1682951086-0-0-5d5e71e1b4169ad8c53473a1144aeab8')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8093/live/HD_621c81ac-857c-1bc0-c2b5-a03efa3b7722.m3u8?auth_key=1682952183-0-0-07ef9fc2d6e64c252cafaa3d79301506')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8196/live/HD_f21f18e1-2e5c-11eb-912a-0242ac110004.m3u8?auth_key=1682952348-0-0-fd9b7b3aee5e0ab19f5d18ed8eb17950')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8094/live/HD_386ca91a-4b77-11eb-82fc-3cd2e55e0a30.m3u8?auth_key=1682955477-0-0-c28936307a4f59a8564434cbe739833d')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8097/live/HD_7b35be4d-4d75-4982-bbc9-0bb1e09e8ff0.m3u8?auth_key=1682955567-0-0-72f269a3ac56c49a2e2a71eb5cbb67a7')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8098/live/HD_809e0561-f846-436f-8614-dbce1572f1df.m3u8?auth_key=1682955716-0-0-83bbde4993216286d61e0cff67463f1f')
    # detector.detect(r'https://vcsplay.scjtonline.cn:8182/live/HD_386fa90a-4b77-11eb-82fc-3cd2e55e0a30.m3u8?auth_key=1682977327-0-0-2d72fef7e5d790e404dbb71bf925703b')
    detector.detect(r'https://vcsplay.scjtonline.cn:8195/live/HD_c53f71b3-4a6d-11eb-8edc-3cd2e55e088c.m3u8?auth_key=1683000448-0-0-cefa61b2b4a77cdf9bff5b11636b9c4e')