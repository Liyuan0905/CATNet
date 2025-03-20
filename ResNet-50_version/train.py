import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
from data import get_confidence, get_loader, get_loader_psd, get_loader_rgbd, get_loader_weak, SalObjDatasetRGB, test_dataset, get_loader_trip
# from data_semi import SalImgDatasetRGB, SalGTDatasetRGB, SalObjDatasetRGB

from img_trans import scale_trans
from config import param as option
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import AvgMeter, set_seed, visualize_all, visualize_all3, visualize_all4, adjust_lr
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from img_trans import rot_trans, scale_trans
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mse_loss = torch.nn.MSELoss(reduction='sum')
# mse_loss = torch.nn.MSELoss(reduction='sum')
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
kl_div = torch.nn.KLDivLoss(reduction='none')



def energy(score):
    if option['e_energy_form'] == 'tanh':
        energy = F.tanh(score.squeeze())
    elif option['e_energy_form'] == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif option['e_energy_form'] == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy

def get_optim(option, params):
    optimizer = torch.optim.Adam(params, option['lr'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler


def get_optim_ebm(option, params):
    optimizer = torch.optim.Adam(params, option['lr_ebm'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])
    return optimizer, scheduler


def sample_p_0(n=option['batch_size'], sig=option['e_init_sig']):
    return sig * torch.randn(*[n, option['latent_dim'], 12, 12]).to(device)


def train_one_epoch(pure_model, optimizer, proxy_model, pm_optimizer, train_loader, un_train_loader, loss_fun):
    pure_model.train()
    proxy_model.train()
    
    pure_loss_record, proxy_loss_record = AvgMeter(), AvgMeter()
    print('Pure_Model Learning Rate: {:.2e}'.format(optimizer.param_groups[0]['lr']))
    
    
    #  1. Training with the Labeled Datasets 
    sal_count = 0          
    pro_count = 0          
    progress_bar_labeled = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar_labeled):
        for rate in size_rates:
            optimizer.zero_grad()
            pm_optimizer.zero_grad()
            
            images, images_ch, gts, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
            
            ref_pre_0 = pure_model(images)
            ref_pre_0_sig = torch.sigmoid(ref_pre_0)

            ref_pre_1 = pure_model(images_ch)
            ref_pre_1_sig = torch.sigmoid(ref_pre_1)
            

            pro_pre_0 = proxy_model(images)
            pro_pre_0_sig = torch.sigmoid(pro_pre_0)

            pro_pre_1 = proxy_model(images_ch)
            pro_pre_1_sig = torch.sigmoid(pro_pre_1)
            
            loss = cal_loss(ref_pre_0, gts, loss_fun) + cal_loss(ref_pre_1, gts, loss_fun) + (mse_loss(ref_pre_0_sig, ref_pre_1_sig)).mean()
            loss.backward()
            optimizer.step()
            

            loss_pm = cal_loss(pro_pre_0, gts, loss_fun) + cal_loss(pro_pre_1, gts, loss_fun) + (mse_loss(pro_pre_0_sig, pro_pre_1_sig)).mean()
            loss_pm.backward()
            pm_optimizer.step()

            pro_dis = (torch.abs(pro_pre_0_sig - pro_pre_1_sig)).mean()
            pro_count += pro_dis
            
            sal_dis = (torch.abs(ref_pre_0_sig - ref_pre_1_sig)).mean()
            sal_count += sal_dis
            
            
            visualize_all3(torch.sigmoid(ref_pre_0), torch.sigmoid(pro_pre_0), gts, option['log_path'])
            if rate == 1:
                pure_loss_record.update(loss.data, option['batch_size'])
                proxy_loss_record.update(loss_pm.data, option['batch_size'])
                
        progress_bar_labeled.set_postfix(gen_loss=f'{pure_loss_record.show():.5f}', proxy_loss=f'{proxy_loss_record.show():.5f}') 

    avg_pro_dis = pro_count / len(train_loader)
    avg_sal_dis = sal_count / len(train_loader)
    print("avg_pro_dis:", avg_pro_dis)
    print("avg_sal_dis:", avg_sal_dis)

    #  2. Training with the UnLabeled Datasets by using proxy model        
    if (avg_pro_dis < 0.01) and (avg_pro_dis < avg_sal_dis):
        print("Training with unlabeled images by using proxy model.")
        uspv_sal_count = 0          
        uspv_pro_count = 0  
        progress_bar_un_labeled = tqdm(un_train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
        for i, pack in enumerate(progress_bar_un_labeled):
            for rate in size_rates:
                optimizer.zero_grad()
                
                images, images_ch, _, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
                
                ref_pre_0 = pure_model(images)
                ref_pre_0_sig = torch.sigmoid(ref_pre_0)

                ref_pre_1 = pure_model(images_ch)
                ref_pre_1_sig = torch.sigmoid(ref_pre_1)

                with torch.no_grad():
                    prox_pre_0 = proxy_model(images)
                    prox_pre_0_sig = torch.sigmoid(prox_pre_0) 

                    prox_pre_1 = proxy_model(images_ch)
                    prox_pre_1_sig = torch.sigmoid(prox_pre_1)   
                
                loss = 0.5*(cal_loss(ref_pre_0, prox_pre_0_sig, loss_fun) + cal_loss(ref_pre_1, prox_pre_0_sig, loss_fun) \
                            + (mse_loss(ref_pre_1_sig, ref_pre_0_sig)).mean())
                loss.backward()
                optimizer.step()
                
                pro_dis = (torch.abs(prox_pre_0_sig - prox_pre_1_sig)).mean()
                uspv_pro_count += pro_dis
            
                sal_dis = (torch.abs(ref_pre_0_sig - ref_pre_1_sig)).mean()
                uspv_sal_count += sal_dis

                visualize_all3(ref_pre_0_sig, ref_pre_1_sig, prox_pre_0_sig, option['log_path'])
                if rate == 1:
                    pure_loss_record.update(loss.data, option['batch_size'])
                    
            progress_bar_un_labeled.set_postfix(gen_loss=f'{pure_loss_record.show():.5f}') 
        
        uspv_avg_pro_dis = uspv_pro_count / len(un_train_loader)
        uspv_avg_sal_dis = uspv_sal_count / len(un_train_loader)
        print("uspv_avg_pro_dis:", uspv_avg_pro_dis)
        print("uspv_avg_sal_dis:", uspv_avg_sal_dis)
        
        if (uspv_avg_sal_dis < uspv_avg_pro_dis):
            print("Next, updating the proxy model.")
            progress_bar_un_labeled = tqdm(un_train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
            for i, pack in enumerate(progress_bar_un_labeled):
                for rate in size_rates:
                    pm_optimizer.zero_grad()
                    
                    images, images_ch, _, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
                    
                    prox_pre_0 = proxy_model(images)
                    prox_pre_0_sig = torch.sigmoid(prox_pre_0)    
                    
                    prox_pre_1 = proxy_model(images_ch)
                    prox_pre_1_sig = torch.sigmoid(prox_pre_1)    
                    
                    with torch.no_grad():
                        ref_pre_0 = pure_model(images)
                        ref_pre_0_sig = torch.sigmoid(ref_pre_0)

                        
                    loss_pm = 0.5*(cal_loss(prox_pre_0, ref_pre_0_sig, loss_fun) + cal_loss(prox_pre_1, ref_pre_0_sig, loss_fun) \
                                + (mse_loss(prox_pre_1_sig, prox_pre_0_sig)).mean())
                    loss_pm.backward()
                    pm_optimizer.step()
                    

                    visualize_all3(prox_pre_0_sig, prox_pre_1_sig, ref_pre_0_sig, option['log_path'])
                    if rate == 1:
                        proxy_loss_record.update(loss_pm.data, option['batch_size'])
                progress_bar_un_labeled.set_postfix(proxy_loss=f'{proxy_loss_record.show():.5f}') 

    elif (avg_sal_dis < 0.01) and (avg_sal_dis < avg_pro_dis):
        print("Training with unlabeled images by using the saliency model.")
        uspv_sal_count = 0          
        uspv_pro_count = 0  
        progress_bar_un_labeled = tqdm(un_train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
        for i, pack in enumerate(progress_bar_un_labeled):
            for rate in size_rates:
                pm_optimizer.zero_grad()
                
                images, images_ch, _, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
                
                prox_pre_0 = proxy_model(images)
                prox_pre_0_sig = torch.sigmoid(prox_pre_0)    
                
                prox_pre_1 = proxy_model(images_ch)
                prox_pre_1_sig = torch.sigmoid(prox_pre_1)    
                
                with torch.no_grad():
                    ref_pre_0 = pure_model(images)
                    ref_pre_0_sig = torch.sigmoid(ref_pre_0)

                    ref_pre_1 = pure_model(images_ch)
                    ref_pre_1_sig = torch.sigmoid(ref_pre_1)

                    
                loss_pm = 0.5*(cal_loss(prox_pre_0, ref_pre_0_sig, loss_fun) + cal_loss(prox_pre_1, ref_pre_0_sig, loss_fun) \
                            + (mse_loss(prox_pre_1_sig, prox_pre_0_sig)).mean())
                loss_pm.backward()
                pm_optimizer.step()
                

                pro_dis = (torch.abs(prox_pre_0_sig - prox_pre_1_sig)).mean()
                uspv_pro_count += pro_dis
            
                sal_dis = (torch.abs(ref_pre_0_sig - ref_pre_1_sig)).mean()
                uspv_sal_count += sal_dis

                visualize_all3(prox_pre_0_sig, prox_pre_1_sig, ref_pre_0_sig, option['log_path'])
                if rate == 1:
                    proxy_loss_record.update(loss_pm.data, option['batch_size'])
            progress_bar_un_labeled.set_postfix(proxy_loss=f'{proxy_loss_record.show():.5f}') 
        
        uspv_avg_pro_dis = uspv_pro_count / len(un_train_loader)
        uspv_avg_sal_dis = uspv_sal_count / len(un_train_loader)
        print("uspv_avg_pro_dis:", uspv_avg_pro_dis)
        print("uspv_avg_sal_dis:", uspv_avg_sal_dis)
        
        if (uspv_avg_pro_dis < uspv_avg_sal_dis):
            print("Next, updating the saliency model.")        
            progress_bar_un_labeled = tqdm(un_train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
            for i, pack in enumerate(progress_bar_un_labeled):
                for rate in size_rates:
                    optimizer.zero_grad()
                    
                    images, images_ch, _, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
                    
                    ref_pre_0 = pure_model(images)
                    ref_pre_0_sig = torch.sigmoid(ref_pre_0)

                    ref_pre_1 = pure_model(images_ch)
                    ref_pre_1_sig = torch.sigmoid(ref_pre_1)

                    with torch.no_grad():
                        prox_pre_0 = proxy_model(images)
                        prox_pre_0_sig = torch.sigmoid(prox_pre_0)    
                    
                    loss = 0.5*(cal_loss(ref_pre_0, prox_pre_0_sig, loss_fun) + cal_loss(ref_pre_1, prox_pre_0_sig, loss_fun) \
                                + (mse_loss(ref_pre_1_sig, ref_pre_0_sig)).mean())
                    loss.backward()
                    optimizer.step()
                    

                    visualize_all3(ref_pre_0_sig, ref_pre_1_sig, prox_pre_0_sig, option['log_path'])
                    if rate == 1:
                        pure_loss_record.update(loss.data, option['batch_size'])
                        
                progress_bar_un_labeled.set_postfix(gen_loss=f'{pure_loss_record.show():.5f}') 
            

    adjust_lr(optimizer, option['lr'], epoch, option['decay_rate'], option['decay_epoch'])
    adjust_lr(pm_optimizer, option['lr'], epoch, option['decay_rate'], option['decay_epoch'])

    return pure_model, pure_loss_record, proxy_model, proxy_loss_record


if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    pure_model, proxy_model = get_model(option)
    optimizer, scheduler = get_optim(option, pure_model.parameters())
    pm_optimizer, pm_scheduler = get_optim(option, proxy_model.parameters())
    
    task_name = option['task']
    
    train_loader,  un_train_loader = get_loader(image_root=option['image_root'], gt_root=option['gt_root'], 
                                                ratio=option['partial_ratio'], batchsize=option['batch_size'], trainsize=option['trainsize'])

    # train_z = torch.FloatTensor(train_set_size, option['latent_dim']).normal_(0, 1).cuda()
    size_rates = option['size_rates']  # multi-scale training
    writer = SummaryWriter(option['log_path'])
    for epoch in range(1, (option['epoch']+1)):
        pure_model, pure_loss_record, proxy_model, proxy_loss_record = train_one_epoch(pure_model, optimizer, proxy_model, pm_optimizer, \
                                                                                                 train_loader, un_train_loader, loss_fun)
        
    
        writer.add_scalar('Pure_Model_init loss', pure_loss_record.show(), epoch)
        writer.add_scalar('Pure_Model_init_lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        
        writer.add_scalar('Proxy_Model_init loss', proxy_loss_record.show(), epoch)
        writer.add_scalar('Proxy_Model_init_lr', pm_optimizer.param_groups[0]['lr'], epoch)
        pm_scheduler.step()
        
        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(pure_model.state_dict(), save_path + '/{:d}'.format(epoch) + '_model.pth')
            torch.save(proxy_model.state_dict(), save_path + '/{:d}'.format(epoch) + '_proxy_model.pth')
            
