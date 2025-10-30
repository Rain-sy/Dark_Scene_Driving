import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_boxes
from yolov7.utils.datasets import LoadImages

class run_yolov7:
    def __init__(self, weights, device='cpu'):
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()

    def detect(self, img_path, conf_thres=0.25, iou_thres=0.45):
        dataset = LoadImages(img_path, img_size=640)
        results = []
        for path, img, im0s, vid_cap, s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                    results.append(det.cpu().numpy())
        return results
