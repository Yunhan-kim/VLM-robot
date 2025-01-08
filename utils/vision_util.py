import torch
import cv2
import numpy as np
from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops

from segment_anything.segment_anything import build_sam, SamPredictor


class VisionBase:
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
    
    def obb_predict(self, image_path, text_prompt, box_threshold = 0.4, text_threshold = 0.5):
        # bounding box
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model = self.groundingdino_model, 
            image = image, 
            caption = text_prompt, 
            box_threshold = box_threshold, 
            text_threshold = text_threshold,
            device = self.device
            )
        
        # segmentation
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
        
        obb_pred = []
        for i in range(masks.shape[0]):
            obb = self.get_oriented_bounding_box(masks[i][0])
            obb_pred.append([phrases[i], logits[i].tolist(), boxes_xyxy[i].tolist(), obb])

        return obb_pred
        