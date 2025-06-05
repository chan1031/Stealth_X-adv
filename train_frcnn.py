import os
import time
import random
import argparse
import warnings

'''
이 코드는 지정한 모델을 데이터셋에 맞게 훈련하는 코드
(적대적 객체 생성 코드는 아님)
'''

#경고 메시지 무시
warnings.filterwarnings("ignore")

import torch
import torch.utils.data as data
import numpy as np

from data.dataset import RCNNDetectionDataset, RCNNAnnotationTransform
from data import config
from model import FasterRCNNVGG16, FasterRCNNTrainer

# os.environ["NCCL_DEBUG"] = "INFO"

#문자열을 True or False로 변환하는 함수
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")

#랜덤 시드를 고정하는 함수, 실행할때 마다 같은 결과가 나오도록 함
def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#명령어 인자 설정
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for the experiments')
parser.add_argument('--dataset', default="OPIXray", type=str, 
                    choices=["OPIXray", "HiXray", "XAD"], help='Dataset name')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--transfer', default=None, type=str,
                    help='Checkpoint state_dict file to transfer from')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1*(1e-3), type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default=None, type=str, required=True,
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

fix_seed(args.seed)

#기본적으로 텐서를 Float 타입으로 설정
torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available(): #현재 CUDA를 사용할 수 있는 환경인지
    if not args.cuda: #CUDA가 사용 가능한데 args로 CUDA 설정을 안한 경우
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
#현재 시간
start_time = time.strftime ('%Y-%m-%d_%H-%M-%S')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder, exist_ok=True)

#훈련 함수
def train():
    print(f'Training for {args.dataset} model...')  
    #데이터셋 종류 설정
    if args.dataset == "OPIXray":
        data_info = config.OPIXray_train
    elif args.dataset == "HiXray":
        data_info = config.HiXray_train
    elif args.dataset == "XAD":
        data_info = config.XAD_train
    cfg = config.FasterRCNN #config 설정을 Faster RCNN으로 설정 cfg는 딕셔너리

    # modify cfg, 인자로 입력한 값으로 설정
    cfg['lr'] = args.lr #학습률 설정 
    cfg['weight_decay'] = args.weight_decay #weight_decay L2정규화 설정
    num_classes = len(data_info["model_classes"]) + 1 #클래스의 개수 설정, +1은 배경
    #객체 탐지용 데이터셋을 로드하는 클래스
    dataset = RCNNDetectionDataset(root=data_info["dataset_root"], #데이터셋이 어디에 있는지
                               model_classes=data_info["model_classes"],#모델의 클래스에는 어떠한 것이 있는지
                               image_sets=data_info["imagesetfile"], #이미지에 관한 .txt 파일이 어디에 있는지
                               target_transform=RCNNAnnotationTransform(data_info["model_classes"]), #바운딩 박스좌표를 반환
                               phase='train')
    

    #모델을 불러옴
    frcnn_net = FasterRCNNVGG16(cfg, num_classes - 1, transfer=args.transfer) #transfer로 입력한 경로의 모델을 불러옴
    trainer = FasterRCNNTrainer(frcnn_net, cfg, num_classes).cuda()

    '''
    #모델을 불러옴
    frcnn_net = FasterRCNNVGG16(cfg, num_classes - 1, transfer=args.transfer) #transfer로 입력한 경로의 모델을 불러옴
    # GPU가 2개 이상이면 DataParallel 사용
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        frcnn_net = torch.nn.DataParallel(frcnn_net)
    frcnn_net = frcnn_net.cuda()  # GPU로 이동
    trainer = FasterRCNNTrainer(frcnn_net.module, cfg, num_classes).cuda()
    '''
    
    print(frcnn_net)

    #nn.module의 학습모드로 변경
    trainer.train()
    # loss counters
    epoch = 0
    print('Loading dataset...')

    print('Training SSD on', args.dataset) #오타인듯?
    print('Using the specified args:')
    print(args)
    #데이터 로드 (batch_size 존재)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True)
    #에포크를 200번동안 반복
    for epoch in range(200):
        trainer.reset_meters() #이전 에포크의 지표를 초기화 (손실값,정확도)
        loss_cnt = 0
        for ii, (img, bbox_, label_, scale, ids) in enumerate(data_loader): #ii는 인덱스
            #enumerate는 for문에서 여러 key들을 사용하기 위해 사용
            '''
            img: 변환된 이미지 텐서
            bbox: 바운딩 박스 좌표
            label: 클래스 라벨
            scale: 이미지의 크기 정보
            ids: 이미지의 위치
            '''
            # print(img.shape, bbox_, label_, scale, ids)
            scale = scale.item()
            # print(ids)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda() #.cuda()는 데이터를 GPU로 보냄
            losses = trainer.train_step(img, bbox, label, scale) #손실값을 구함
            loss_cnt += losses.total_loss.item()
            if (ii + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, iter {ii+1}/{len(data_loader)}, losses: {loss_cnt / 50}")
                loss_cnt = 0

        # if (epoch + 1) % 1 == 0:
        torch.save(frcnn_net.state_dict(), args.save_folder + '/model_' +
                    repr(epoch+1) + '.pth')

train()