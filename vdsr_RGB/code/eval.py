import torch as t
from PIL import Image
from torchvision.transforms import ToTensor
from math import log10

import argparse, os

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_1.pth", type=str, help="model path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
cuda = opt.cuda

model = t.load(opt.model, map_location=lambda storage, loc: storage)["model"]

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not t.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = model.cuda()
else:
    model = model.cpu()
    
device = t.device('cuda' if cuda else 'cpu')
            
root = '../data/Set5'            
image_list = ['woman_GT','head_GT','butterfly_GT','bird_GT','baby_GT']

count = 5
scales = [2,4]
model.eval()

def im2tensor(im):
    return ToTensor()(im).unsqueeze(0).to(device)

def psnr(im_GT, im_bic):
    with t.no_grad():
        input = im2tensor(im_bic)
        label = im2tensor(im_GT)
        predict = model(input)
        mse = t.nn.MSELoss()(predict, label)
    return 10*log10(1. / mse.item())

for scale in scales:
    avg_psnr = 0.
    for im_name in image_list:
        im_GT_path = os.path.join(root, im_name + '.bmp')
        im_bic_path = os.path.join(root, im_name + '_scale_{}.bmp'.format(scale))
        
        im_GT = Image.open(im_GT_path)
        im_bic = Image.open(im_bic_path)
        
        avg_psnr += psnr(im_GT, im_bic)
        
    print("Scale=", scale)
    print("PSNR_predicted=", avg_psnr/count)
        
        

        
        
        
