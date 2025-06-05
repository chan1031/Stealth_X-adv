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

def load_groove_coords(txt_path):
    """
    txt 파일에서 'v x y z' 형태의 홈 정점 좌표를 파싱하여 
    (M, 3) shape의 텐서로 반환
    """
    
    coords = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if tokens[0] == 'v':
                x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                coords.append([x, y, z])
    # 텐서로 변환
    groove_coords = torch.tensor(coords, dtype=torch.float32)
    return groove_coords

def create_groove_mask(vertices, groove_coords, eps=1e-6):
    """
    vertices: (N, 3) - 열쇠 전체 정점 (OBJ에서 로드)
    groove_coords: (M, 3) - txt 파일에서 불러온 홈 정점 좌표
    eps: float - 거리 임계값 (정확도)
    
    반환: groove_mask: (N,) bool 텐서
          True면 해당 정점이 '홈' 영역
    """
    N = vertices.shape[0]
    groove_mask = torch.zeros(N, dtype=torch.bool, device=vertices.device)

    for gc in groove_coords:
        # vertices와 gc의 L2 거리 계산
        dist = torch.norm(vertices - gc, dim=1)  # shape: (N,)
        idx = torch.argmin(dist)  # 가장 가까운 정점 인덱스
        if dist[idx] < eps:
            groove_mask[idx] = True  # 이 정점을 홈 영역으로 표시

    return groove_mask

vertices, faces = renderer.load_from_file("/home/skku/Desktop/Adversarial_Example/X-adv/objs/simple_door_key2.obj")
groove_coords = load_groove_coords("groove_points.txt") 
groove_mask = create_groove_mask(vertices, groove_coords, eps=1e-6)

print(groove_mask)

group = []
group.append(vertices.unsqueeze(0))


print(group)
print(groove_coords)

