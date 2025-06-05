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


parser = argparse.ArgumentParser(description="X-ray adversarial attack.")
# for model
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for the experiments')
parser.add_argument("--ckpt_path", default="./ckpt/OPIX.pth", type=str, 
                    help="the checkpoint path of the model")
# for data
parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray"], help='Dataset name')
parser.add_argument("--phase", default="test", type=str, 
                    help="the phase of the X-ray image dataset")
parser.add_argument("--batch_size", default=256, type=int, 
                    help="the batch size of the data loader")
parser.add_argument("--num_workers", default=8, type=int, 
                    help="the number of workers of the data loader")

parser.add_argument("--pixel_material", default="iron", type=str, choices=["iron", "plastic", "aluminum", "iron_fix"],
                    help="the material of patch, which decides the color of patch")  #적대적 객체의 재질 지정   

parser.add_argument("--n_pixels", default=10, type=int,
                    help="number of pixels to modify in each attack")  # 수정할 픽셀 수

parser.add_argument("--brightness_factor", default=1.0, type=float,
                    help="brightness adjustment factor for the pixel color")  # 밝기 조정 계수

parser.add_argument("--lr", default=0.01, type=float,  #학습률
                    help="the learning rate of attack")
parser.add_argument("--num_iters", default=3000, type=int,
                    help="number of iterations for the attack")  # 반복 횟수
parser.add_argument("--save_path", default="./few_pixel_results", type=str,
                    help="the save path of adversarial examples")
parser.add_argument("--lambda_roi", default=5.0, type=float,
                    help="weight for ROI classification loss (highest priority)")
parser.add_argument("--lambda_rpn", default=2.0, type=float,
                    help="weight for RPN classification loss (second priority)")
parser.add_argument("--weight_decay", default=0.1, type=float,
                    help="decay factor for loss weights over iterations")

args = parser.parse_args()
args.save_path = os.path.join(args.save_path, f"{args.dataset}/{args.pixel_material}", "FasterRCNN")


def get_color_adjustment(category):
    """
    색상 범주에 따른 HSV 조정값 반환
    """
    adjustments = {
        "original": {"s": 1.0, "v": 1.0},    # 기본 색상
        "dark": {"s": 2.2, "v": 0.7},        # 진한 색상
        "very_dark": {"s": 2.5, "v": 0.5},   # 매우 진한 색상
        "light": {"s": 0.7, "v": 1.3},       # 얕은 색상
        "very_light": {"s": 0.5, "v": 1.5}   # 매우 얕은 색상
    }
    return adjustments.get(category, adjustments["original"])

#random attack
def random_attack(images, bboxes, labels, net, trainer):
    net.phase = "train"
    images = [e.cuda() for e in images]
    bboxes = [b.cuda() for b in bboxes]
    labels = [l.cuda() for l in labels]

    for i in range(len(images)):
        bbox = bboxes[i][0]  # 첫 번째 객체의 bbox 사용
        
        # 현재 이미지 크기
        current_h, current_w = images[i].shape[1:]  # [300, 385]
        
        # bbox 좌표를 현재 이미지 크기에 맞게 조정 (x와 y 좌표 바꿈)
        y1, x1, y2, x2 = bbox  # x와 y 좌표를 바꿔서 받음
        x1 = max(0, min(int(x1), current_w-1))
        y1 = max(0, min(int(y1), current_h-1))
        x2 = max(0, min(int(x2), current_w-1))
        y2 = max(0, min(int(y2), current_h-1))
        
        # random few pixel attack
        for _ in range(100):  # 100개의 픽셀 수정
            random_x = int(random.uniform(x1, x2))
            random_y = int(random.uniform(y1, y2))
            
            # 해당 픽셀을 IRON 재질 색상으로 수정
            # 1x1 크기의 텐서 생성 (깊이 이미지 형태)
            depth_pixel = torch.ones((1, 1, 1, 1), device=images[i].device)
            # simulate 함수를 사용하여 IRON 재질 색상 생성
            color_pixel, _ = renderer.simulate(depth_pixel, material="iron")
            
            # RGB 값 확인 (0-1 범위)
            r = color_pixel[0, 2, 0, 0].item()  # Red
            g = color_pixel[0, 1, 0, 0].item()  # Green
            b = color_pixel[0, 0, 0, 0].item()  # Blue
            
            # RGB 값을 0-255 범위로 변환
            r_255 = int(r * 255)
            g_255 = int(g * 255)
            b_255 = int(b * 255)
            
            print(f"RGB values (0-255): R={r_255}, G={g_255}, B={b_255}")
            
            # 생성된 색상을 이미지에 적용
            images[i][:, random_y, random_x] = color_pixel[0, :, 0, 0]

    return images

'''
faster rcnn loss 함수
1.L_rpn_cls: rpn의 각 anchor가 물체를 포함하는지 여부를 검사
2.L_rpn_loc: RPN의 bounding box 위치 조정
3.L_roi_cls: ROI가 어떤 클래스에 속하는지 분류
4.L_roi_loc: ROI의 bounding box 조정 (클래스별)

'''

#few pixel attack
def attack(images, bboxes, labels, net, trainer):
    net.phase = "train"
    images = [e.cuda() for e in images]
    bboxes = [b.cuda() for b in bboxes]
    labels = [l.cuda() for l in labels]

    #이미지를 한개씩 처리
    for i in range(len(images)):
        bbox = bboxes[i][0]  # 첫 번째 객체의 bbox 사용
        current_h, current_w = images[i].shape[1:]
        y1, x1, y2, x2 = bbox
        x1 = max(0, min(int(x1), current_w-1))
        y1 = max(0, min(int(y1), current_h-1))
        x2 = max(0, min(int(x2), current_w-1))
        y2 = max(0, min(int(y2), current_h-1))
        
        # 재질에 따른 색상 설정
        # 1x1 크기의 깊이 이미지 생성 (0.5로 초기화)
        depth_pixel = torch.ones((1, 1, 1, 1), device=images[i].device) * 0.5
        # simulate 함수를 사용하여 재질 색상 생성
        base_color, _ = renderer.simulate(depth_pixel, material=args.pixel_material)
        base_color = base_color[0, :, 0, 0]  # (3,) 텐서로 변환
        
        # bbox 영역만 추출하여 해당 부분만 변화하게 설정
        bbox_region = images[i][:, y1:y2+1, x1:x2+1].clone()
        bbox_region.requires_grad_(True)
        
        #원본 이미지
        images[i] = images[i].clone().detach()

        # 반복 횟수 설정
        for iter in range(args.num_iters):
            #현재 이미지에 대한 손실함수 가중치 처리
            current_lambda_roi = args.lambda_roi * (1.0 - args.weight_decay * iter / args.num_iters)
            current_lambda_rpn = args.lambda_rpn * (1.0 - args.weight_decay * iter / args.num_iters)
            
            #이미지에 대한 그라디언트 초기화
            if bbox_region.grad is not None:
                bbox_region.grad.zero_()
            trainer.reset_meters()

            #이미지에 대한 손실함수 계산
            loss_cls = trainer.forward(images[i].unsqueeze(0), bboxes[i].unsqueeze(0), labels[i].unsqueeze(0), 1.0) #이미지를 모델에 넣어서 손실값을 획득
            #공격을 위한 손실함수 정의 (roi_cls_loss와 rpn_cls_loss를 가중치를 줘서 공격)
            L_attack = (current_lambda_roi * loss_cls.roi_cls_loss + 
                       current_lambda_rpn * loss_cls.rpn_cls_loss)
            
            #적대적 공격이므로 부호를 반대화
            loss_adv = -L_attack
            
            #기울기를 계산해서 image 픽셀에 대한 기울기 값을 계산한다.
            loss_adv.backward()
            
            #기울기 값이 존재한다면 공격진행 ㄱ
            if images[i].grad is not None and images[i].grad.abs().sum() > 0:
                #이미지 전체 픽셀의 기울기 값을 가져옴. 단, 절대값을 넣어서 크기만을 가져옴
                grad = images[i].grad.abs()
                #RGB전체 3채널의 기울기 값을 더해서 하나의 픽셀에 대한 기울기 값을 계산
                grad_magnitude = grad.sum(dim=0)
                
                #이미지와 같은 크기의 0으로 채워진 텐서 생성
                grad_magnitude_bbox = torch.zeros_like(grad_magnitude)
                #핵심 bbox영역만 복사
                grad_magnitude_bbox[y1:y2+1, x1:x2+1] = grad_magnitude[y1:y2+1, x1:x2+1]
                #1차원으로 변환
                flat_grad = grad_magnitude_bbox.view(-1)
                
                #기울기 값이 존재한다면 공격진행
                if flat_grad.max() > 0:
                    # 전체 반복에서 n_pixels개의 픽셀만 선택
                    if iter == 0:  # 첫 번째 반복에서만 픽셀 선택
                        _, indices = torch.topk(flat_grad, min(args.n_pixels, flat_grad.numel()))
                        selected_y = indices // current_w
                        selected_x = indices % current_w
                        selected_pixels = list(zip(selected_y, selected_x))
                    
                    # 선택된 픽셀들만 변형
                    for y, x in selected_pixels:
                        #이미지 3 채널 (R,G,B)에 대한 기울기 값을 가져옴
                        pixel_grad = images[i].grad[:, y, x]
                        #기울기 값의 부호를 가져옴
                        grad_direction = torch.sign(pixel_grad)
                        #기울기 값의 크기를 조정
                        brightness_adjustment = 1.0 - args.lr * grad_direction.mean() #이미 부호를 바꾸었으므로 기울기의 반대방향으로 계산이 된다.
                        #밝기 조정을 50~150%로 유지
                        brightness_adjustment = torch.clamp(brightness_adjustment, 0.5, 1.5)
                        #밝기 적용, 기본 색상에 밝기 조정 값을 곱해서 새로운 색상 생성
                        adjusted_color = base_color * brightness_adjustment
                        adjusted_color = torch.clamp(adjusted_color, 0.0, 1.0)
                         
                        #이미지에 적용     
                        with torch.no_grad():
                            images[i][:, y, x] = adjusted_color
                
                print(f"Iteration {iter+1}/{args.num_iters}, Loss: {loss_adv.item():.4f}, "
                      f"ROI Cls: {loss_cls.roi_cls_loss.item():.4f} (w={current_lambda_roi:.2f}), "
                      f"RPN Cls: {loss_cls.rpn_cls_loss.item():.4f} (w={current_lambda_rpn:.2f})")
            else:
                print(f"Iteration {iter+1}/{args.num_iters}: No gradients computed or zero gradients")
                break

            # === 탐지 여부 확인 ===
            img_np = images[i].detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            if isinstance(net, nn.DataParallel):
                bboxes_pred, labels_pred, scores_pred = net.module.predict([np.transpose(img_np, (2, 0, 1))], sizes=[img_np.shape[:2]])
            else:
                bboxes_pred, labels_pred, scores_pred = net.predict([np.transpose(img_np, (2, 0, 1))], sizes=[img_np.shape[:2]])
            
            if len(bboxes_pred[0]) == 0:
                print(f"공격 성공! {iter+1}번째 반복에서 탐지되지 않음.")
                # 공격이 성공하면 즉시 이미지 저장
                shape = images[i].shape[1:]
                shape = [int(shape[0] / scales[i]), int(shape[1] / scales[i])]
                save_img(os.path.join(img_path, img_ids[i] + ".png"), images[i] * 255, shape)
                print(f"Saved image {img_ids[i]} (attack success)")
                break
            
            # 마지막 반복에서도 이미지 저장
            if iter == args.num_iters - 1:
                print(f"공격 실패: 최대 반복 횟수({args.num_iters})에 도달")
                shape = images[i].shape[1:]
                shape = [int(shape[0] / scales[i]), int(shape[1] / scales[i])]
                save_img(os.path.join(img_path, img_ids[i] + ".png"), images[i] * 255, shape)
                print(f"Saved image {img_ids[i]} (attack failed)")

    return images

'''
이미지를 저장하는 함수
'''
def save_img(path, img_tensor, shape):
    img_tensor = img_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.transpose(1, 2, 0)
    img = cv2.resize(img, (shape[1], shape[0]))
    cv2.imwrite(path, img)
    
if __name__ == "__main__":
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_test
    elif args.dataset == "HiXray":
        data_info = config.HiXray_test
    
    #클래스의 개수
    num_classes = len(data_info["model_classes"]) + 1

    #모델 생성
    net = FasterRCNNVGG16(config.FasterRCNN, num_classes - 1)

    state_dict = torch.load("./save/model_200.pth")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    
    net.cuda()
    
    #GPU setting
    gpu_count = torch.cuda.device_count()
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA visible device count:", gpu_count)
    
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs via DataParallel!")
        net = nn.DataParallel(net, device_ids=[0, 1])
    
    #create trainer
    trainer = FasterRCNNTrainer(net, config.FasterRCNN, num_classes).cuda()
    net.eval()
    
    #load dataset
    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], 
                               model_classes=data_info["model_classes"],
                               image_sets=data_info["imagesetfile"], 
                               target_transform=RCNNAnnotationTransform(data_info["model_classes"]), 
                               phase='test')
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=rcnn_detection_collate_attack, pin_memory=True)
    
    num_images = len(dataset)
    
    '''
    저장경로 설정 부분
    '''
    #만약 저장 경로가 없으면 생성
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    #적대적 이미지 저장 경로   
    img_path = os.path.join(args.save_path, "adver_image")
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    '''
    #이미지 한개만 가지고 TEST
    for i, (images, bboxes, labels, scales, img_ids) in enumerate(data_loader):
        if i == 0:  # 첫 번째 배치만 처리
            print("Batch {}/{}...".format(i+1, math.ceil(num_images / args.batch_size)))
            print(img_ids)
            print(bboxes)
            print(labels)
            images =  random_attack(images, bboxes, labels, net, trainer)
            
            # 수정된 이미지 저장
            for t in range(len(images)):
                shape = images[t].shape[1:]
                shape = [int(shape[0] / scales[t]), int(shape[1] / scales[t])]
                save_img(os.path.join(img_path, img_ids[t] + ".png"), images[t] * 255, shape)
            
            break 
    '''
    #attack 함수 호출, 공격진행       
    for i, (images, bboxes, labels, scales, img_ids) in enumerate(data_loader):
        print("Batch {}/{}...".format(i+1, math.ceil(num_images / args.batch_size)))
        images = attack(images, bboxes, labels, net, trainer)
    
    
    
   