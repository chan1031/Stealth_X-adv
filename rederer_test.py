import os
import sys
import cv2
import math
import time
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from model import FasterRCNNVGG16, FasterRCNNTrainer

from data.dataset import RCNNDetectionDataset, rcnn_detection_collate_attack, RCNNAnnotationTransform
from data import config
from utils import stick, renderer, rl_utils

def save_image_as_jpg(image_tensor, save_path):
    """
    Tensor 이미지를 JPG 파일로 저장하는 함수.
    - image_tensor: (C, H, W) 또는 (H, W) 형태의 텐서
    - save_path: 저장할 파일 경로
    """
    image_np = image_tensor.squeeze().cpu().numpy()
    if image_np.ndim == 2:  # Grayscale 이미지
        image_np = (image_np * 255).astype(np.uint8)
        cv2.imwrite(save_path, image_np)
    else:  # RGB 이미지
        image_np = (image_np * 255).astype(np.uint8)
        image_np = image_np.transpose(1, 2, 0)  # (H, W, 3)로 변환
        cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

def convert_obj_to_xray(obj_path, patch_size, material="iron", output_prefix="output"):
    """
    OBJ 파일을 로드하여 X-ray 이미지로 변환 후 JPG로 저장하는 함수.
    
    Parameters:
    - obj_path: 입력 OBJ 파일 경로
    - patch_size: 생성할 이미지의 해상도 (patch_size x patch_size)
    - material: X-ray 시뮬레이션에 사용할 재질 ("iron", "plastic", "aluminum" 등)
    - output_prefix: 저장할 파일 이름 접두사
    """
    # 1. OBJ 파일에서 정점(vertices)과 면(faces) 정보 로드
    vertices, faces = renderer.load_from_file(obj_path)
    
    # 2. 원본 코드와 동일하게 정점 클램핑: 0~1 범위로 정규화
    vertices_clamp = torch.clamp(vertices, 0, 1)
    
    # 3. ball2depth 함수를 사용하여 깊이(Depth) 이미지 생성
    #    결과 텐서 shape: (1, 1, patch_size, patch_size)
    depth_img = renderer.ball2depth(vertices_clamp, faces, patch_size, patch_size).unsqueeze(0).unsqueeze(0)
    
    # 4. simulate 함수를 이용하여 깊이 이미지를 X-ray 이미지로 변환
    #    xray_img의 shape: (1, 3, patch_size, patch_size)
    xray_img, mask = renderer.simulate(depth_img, material)
    
    # 5. 생성된 이미지들을 JPG 파일로 저장
    save_image_as_jpg(depth_img, f"{output_prefix}_depth.jpg")
    save_image_as_jpg(xray_img, f"{output_prefix}_xray.jpg")
    
    print(f"저장 완료: {output_prefix}_depth.jpg, {output_prefix}_xray.jpg")
    return vertices, faces, depth_img, xray_img

# 실행 예제
if __name__ == "__main__":
    obj_file_path = "/home/skku/Desktop/Adversarial_Example/X-adv/objs/simple_door_key2.obj"    # 변환할 OBJ 파일 경로 (사용자 파일 경로로 변경)
    patch_size = 100                # 생성할 이미지 해상도 (예: 50x50)
    material = "iron"              # 사용할 재질 (예: "iron", "plastic",     "aluminum")
    output_prefix = "result"       # 저장 파일 접두사
    
    convert_obj_to_xray(obj_file_path, patch_size, material, output_prefix)
