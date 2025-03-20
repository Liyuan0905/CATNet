import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from data import test_dataset, eval_Dataset
from tqdm import tqdm
# from model.DPT import DPTSegmentationModel
from config import param as option
from model.get_model import get_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable



def eval_mae(loader, cuda=True):
    avg_mae, img_num, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in loader:
            if cuda:
                pred, gt = trans(pred).cuda(), trans(gt).cuda()
            else:
                pred, gt = trans(pred), trans(gt)
            mae = torch.abs(pred - gt).mean()
            if mae == mae: # for Nan
                avg_mae += mae
                img_num += 1.0
        avg_mae /= img_num
    return avg_mae


# Begin the testing process
_, generator = get_model(option)
generator.load_state_dict(torch.load(option['ckpt_save_path']+'/30_model.pth'))
# generator.cuda()
generator.eval()

# ebm_model.load_state_dict(torch.load(option['ckpt_save_path']+'/30_ebm.pth'))
# ebm_model.eval()


test_datasets, pre_root = option['datasets'], option['eval_save_path']

time_list, mae_list = [], []
test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]

save_path_base = pre_root + test_epoch_num + '_ebm/'

# save_path_base = pre_root
# Begin to inference and save masks
print('========== Begin to inference and save masks ==========')
def sample_p_0(n=option['batch_size'], sig=option['e_init_sig']):
    return sig * torch.randn(*[n, option['latent_dim'], 12, 12]).to(device)
index = 0


for dataset in test_datasets:

    save_path_mean = './results_mean/' + dataset + '/'
    if not os.path.exists(save_path_mean):
        os.makedirs(save_path_mean)
   
    image_root = ''
    # depth_root = ''
    # if option['task'] == 'SOD':
    image_root = option['test_dataset_root'] + dataset + '/'
    # elif option['task'] == 'RGBD-SOD':
    #     image_root = option['test_dataset_root'] + dataset + '/RGB/'
    #     depth_root = option['test_dataset_root'] + dataset + '/depth/'

    test_loader = test_dataset(image_root, option['testsize'])
    #for iter in range(9):
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        mean_pred = 0
        # alea_uncertainty = 0

        torch.cuda.synchronize()
        start = time.time()
        
        
        res = generator.forward(image)  # Inference and get the last one of the output list
        mean_pred = mean_pred + torch.sigmoid(res)
        mean_prediction = mean_pred / option['iter_num']
        
        res = F.upsample(mean_prediction, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_mean + name, res)
        

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
    print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))
