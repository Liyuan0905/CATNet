import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
generator,_ = get_model(option)
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

    save_path_mean = './DUTS_TR/' + dataset + '/'
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
        
        # z_e_0 = sample_p_0(n=image.shape[0])
        # z_e_0 = Variable(z_e_0)
        # z5 = z_e_0.detach()
        # # z_e_5.requires_grad = True
        # for kk in range(option['e_l_steps']):
        #     en5 = ebm_model(z_e_5)
        #     z_grad_5 = torch.autograd.grad(en5, z_e_5, grad_outputs=torch.ones_like(en5))[0]
        #     z_e_5.data = z_e_5.data - 0.5 * option['e_l_step_size'] * option['e_l_step_size'] * (z_grad_5 + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z_e_5.data)
        #     z_e_5.data += option['e_l_step_size'] * torch.randn_like(z_e_5).data
        # z5 = z_e_5.detach()  ## z_
        
        res = generator.forward(image)  # Inference and get the last one of the output list
        mean_pred = mean_pred + torch.sigmoid(res)
            # preds = torch.sigmoid(res[0])
            # cur_alea = -1 * preds * torch.log(preds + 1e-8)
            # cur_alea = compute_entropy(preds)
            # alea_uncertainty = alea_uncertainty + cur_alea

        mean_prediction = mean_pred / option['iter_num']
        # alea_uncertainty = alea_uncertainty / option['iter_num']
        # predictive_uncertainty = -1 * mean_prediction * torch.log(mean_prediction + 1e-8)
        # predictive_uncertainty = compute_entropy(mean_prediction)
        # epistemic_uncertainty = predictive_uncertainty - alea_uncertainty

        res = F.upsample(mean_prediction, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_mean + name, res)
        '''
        res = F.upsample(alea_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_aleatoric + name, res)

        res = F.upsample(epistemic_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_epistemic + name, res)

        res = F.upsample(predictive_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.astype(np.uint8)
        res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_predictive + name, res)
        '''

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
    print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))

# # Begin to evaluate the saved masks
# print('========== Begin to evaluate the saved masks ==========')
# for dataset in tqdm(test_datasets):
#     if option['task'] == 'COD':
#         gt_root = option['test_dataset_root'] + dataset + '/GT'
#     else:
#         gt_root = option['test_dataset_root'] + '/GT/' + dataset + '/'
#
#     loader = eval_Dataset(os.path.join(save_path_base, dataset), gt_root)
#     mae = eval_mae(loader=loader, cuda=True)
#     mae_list.append(mae.item())
#
# print('--------------- Results ---------------')
# results = np.array(mae_list)
# results = np.reshape(results, [1, len(results)])
# mae_table = pd.DataFrame(data=results, columns=test_datasets)
# with open(save_path_base+'results.csv', 'w') as f:
#     mae_table.to_csv(f, index=False, float_format="%.4f")
# print(mae_table.to_string(index=False))
# print('--------------- Results ---------------')
