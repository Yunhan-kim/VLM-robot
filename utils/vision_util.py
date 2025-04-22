import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import cv2
import numpy as np
from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops

import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

from segment_anything.segment_anything import build_sam, SamPredictor


class VisionBase:
    def load_sam(self):
        sam_folder = Path(__file__).resolve().parent.parent / "checkpoints"    
        sam_checkpoint = sam_folder / "sam_vit_h_4b8939.pth"
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device = self.device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor    
    
    def get_oriented_bounding_box(self, mask_tensor):
        """
        텐서 형태의 마스크에서 Oriented Bounding Box(회전된 최소 경계 상자)를 찾는 함수.

        Parameters:
        - mask_tensor: PyTorch 텐서 형태의 마스크 (shape: [H, W], 값: 0 또는 1)

        Returns:
        - boxes: 회전된 경계 상자 (각각의 (center, size, angle))
        """
        # PyTorch 텐서를 NumPy 배열로 변환 (CPU로 이동 후 numpy 배열로 변환)
        mask = mask_tensor.cpu().numpy().astype(np.uint8)

        # 마스크에서 컨투어를 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        
        for contour in contours:
            # 각 컨투어에 대해 최소 회전된 경계 상자 계산
            rect = cv2.minAreaRect(contour)
            boxes.append(rect)
        
        return boxes

class GroundingDINO_Vision(VisionBase):
    def __init__(self, device = 'cpu'):
        self.device = device
        self.groundingdino_model = self.load_groundingdino()
        self.sam_predictor = self.load_sam()    

    def load_groundingdino(self):
        groundingdino_folder = Path(__file__).resolve().parent.parent / "checkpoints"    
        groundingdino_checkpoint = groundingdino_folder / "groundingdino_swinb_cogcoor.pth"
        groundingdino_cfg = groundingdino_folder / "GroundingDINO_SwinB_cfg.py"
        model = load_model(groundingdino_cfg, groundingdino_checkpoint)        
        return model        
    
    def obb_predict(self, image_path, text_prompt, box_threshold = 0.4, text_threshold = 0.5):                
        # bounding box
        start = time.time()
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model = self.groundingdino_model, 
            image = image, 
            caption = text_prompt, 
            box_threshold = box_threshold, 
            text_threshold = text_threshold,
            device = self.device
            )
        print("Object detection time(s) :", time.time() - start)
        
        # segmentation
        start = time.time()
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        print("Segmentation time(s) :", time.time() - start)        
        
        start = time.time()
        obb_pred = []
        for i in range(masks.shape[0]):
            obb = self.get_oriented_bounding_box(masks[i][0])
            obb_pred.append([phrases[i], logits[i].tolist(), boxes_xyxy[i].tolist(), obb])
        print("Obb calculation time(s) :", time.time() - start)

        return obb_pred
        
class YOLOWorld_Vision(VisionBase):
    def __init__(self, device = 'cpu'):
        self.device = device
        self.model = YOLOWorld(model_id="yolo_world/s")
        self.sam_predictor = self.load_sam()
    
    def obb_predict(self, image_path, text_prompt, confidence = 0.03):        
        import time
        start = time.time()
        # bounding box
        image_source, image = load_image(image_path)

        results = self.model.infer(image_source, text=text_prompt, confidence=confidence)
        detections = sv.Detections.from_inference(results)
        
        print("time :", time.time() - start)
        # segmentation
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = torch.tensor(detections.xyxy)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        logits = detections.confidence
        phrases = detections.data['class_name']
        
        obb_pred = []
        for i in range(masks.shape[0]):
            obb = self.get_oriented_bounding_box(masks[i][0])
            obb_pred.append([phrases[i], logits[i].tolist(), boxes_xyxy[i].tolist(), obb])

        return obb_pred