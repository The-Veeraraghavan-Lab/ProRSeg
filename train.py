#!/usr/bin/env python

import torch
import matplotlib
matplotlib.use('Agg')

from utils_dataloader import get_dataloader
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(211)

plot_loss_value=[]
plot_loss_value1=[]
plot_loss_value2=[]
plot_loss_value3=[]

import os
import argparse
import time
import numpy as np

import nibabel as nib

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from monai.losses import DiceCELoss
import json
from evaluation import Eval
import random

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True


def save_args(args, filename='args.json'):
    with open(filename, 'w') as file:
        json.dump(vars(args), file, indent=4)


# parse the commandline
parser = argparse.ArgumentParser()
fig = plt.figure()
ax = fig.add_subplot(211)
# data organization parameters
parser.add_argument('--datadir', help='base data directory')
parser.add_argument('--json', help='base data directory')
parser.add_argument('--model_dir', default='models', help='model output directory (default: models)')
parser.add_argument('--cache',  action='store_true', help='Cache training dataset for faster training')
parser.add_argument('--cache_num', type=int, default=200, help='Data of images to be cached into memory (default: 200)')
parser.add_argument('--a_min', default=0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=2000, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')

# training parameters
parser.add_argument('--gpu', default='3', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--workers', type=int, default=4, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--batch_size', type=int, default=3, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--win_sz', type=int, default=3, help='flow downsample factor for integration (default: 3)')
parser.add_argument('--seg_w', type=int, default=3, help='seg weight loss (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--load_model', help='optional model file to initialize with')
parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn_nondet',  action='store_true', help='disable cudnn determinism - might slow down training')
parser.add_argument('--affine',  action='store_true', help='use affine transform while training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int_steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int_downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--inshape', type=int, nargs='+', help='input shape (default 128,192,128)')

# loss hyperparameters
parser.add_argument('--image_loss', default='ncc', help='image reconstruction loss - can be mse or ncc (default: ncc)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')

parser.add_argument('--smooth', type=float, default=150, help='weight of deformation loss (default: 0.01)')
parser.add_argument('--flow_range', type=float, default=5, help='flow range (default: 5)')

parser.add_argument('--flownum', type=int, default=7, help='flow number (default: 7)')
# for output
parser.add_argument('--svdir', type=str, default='train', help='weight of deformation loss (default: 0.01)')
parser.add_argument('--class_list',nargs='+',default=['Liver', 'Lg_Bowel', 'Sm_Bowel', 'Duo_Stomach'],help='List of class names')
#print ('flow_range is ',args.flow_range)
args = parser.parse_args()
smooth_w=args.smooth

bidir = args.bidir
bidir= False
import os

cur_path = os.getcwd()

# Create save directory
args.svdir=args.svdir+'_smooth_'+str(int(args.smooth))+'_range_flow_'+str(int(args.flow_range))+'_batchsize_'+str(args.batch_size)+'_seg_w_'+str(args.seg_w)

sv_folder=cur_path+'/'+args.svdir+'/'
train_tep_sv=sv_folder+'train_tep_sv/'
if not os.path.exists(train_tep_sv):
    os.makedirs(train_tep_sv)

save_args(args, os.path.join(sv_folder, 'args.json'))


print ('*'*40)
print ('smoothness is ',args.smooth)
print ('win_sz is ',args.win_sz)
print ('seg_w is ',args.seg_w)
print ('batch_size is ',args.batch_size)
print ('flow range is ',args.flow_range)

# Input image shape
inshape = args.inshape if args.inshape else (128,192,128)

#class_list = ['Liver', 'Lg_Bowel','Sm_Bowel','Duo_Stomach']
class_list=args.class_list
nlabels=len(class_list)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# Registration Model
if args.load_model:
    model = vxm.networks.VxmDense_3D_LSTM.load(args.load_model, device)
else:
    model = vxm.networks.VxmDense_3D_LSTM_Step_Reg_All_Encoder_LSTM(  
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        range_flow=args.flow_range,
        int_downsize=args.int_downsize
    )
model.to(device)
model.train()

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save


# Registration Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Segmentation Model
model_seg3d = vxm.networks.UNet3D_Seg_LSTM(in_channels=nlabels+1+1,out_channels=nlabels+1,final_sigmoid=False)

# Segmentation Optimizer
optimizer_seg = torch.optim.Adam(model_seg3d.parameters(), lr=args.lr)
model_seg3d=model_seg3d.cuda()


# Registration Image Loss function
if args.image_loss.lower() == 'ncc':
    image_loss_func = vxm.losses.NCC(win=[args.win_sz,args.win_sz,args.win_sz]).loss
elif args.image_loss.lower() == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise NotImplementedError



# prepare deformation loss
smooth_loss_func=vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
grad_loss_func=vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss

#Segmentation loss using cross entropy

seg_loss_cred = DiceCELoss(to_onehot_y=True,
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=0,
                           smooth_dr=1e-1)




structure_loss_cred = vxm.losses.DiceLoss_test_use(num_organ=nlabels)

train_loader_a,val_loader=get_dataloader(args)

total_steps=0       

plt_iternm=[]
flow_num=args.flownum
iter_count=0

flow_ini=torch.zeros(1, 3, *inshape).cuda()
range_flow=1

eval = Eval(train_tep_sv, class_list)


for epoch in range(args.initial_epoch, args.epochs):
    eval.update_epoch((epoch+1))
    train_sv_flag = 0
    print ('running in epoch',epoch)
    schedule_epochs=args.epochs/2
    if epoch>=schedule_epochs:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*(1.0-(epoch-schedule_epochs)/(schedule_epochs)))

    for i_iter, item_ in enumerate(train_loader_a): 
        total_steps=total_steps+1
        if random.uniform(0, 1)>0.5: #randomly swap direction (equal probability)
            Fixed_x=item_['move_img'].float().cuda()
            Fixed_y=item_['move_msk'].float().cuda()
            Moving_x=item_['fixed_img'].float().cuda()
            Moving_y=item_['fixed_msk'].float().cuda()
        else:
            Fixed_x=item_['fixed_img'].float().cuda()
            Fixed_y=item_['fixed_msk'].float().cuda()
            Moving_x=item_['move_img'].float().cuda()
            Moving_y=item_['move_msk'].float().cuda()

        Moving_x=torch.permute(Moving_x, (0, 1, 3,2,4))
        Moving_y=torch.permute(Moving_y, (0, 1, 3,2,4))
        Fixed_y=torch.permute(Fixed_y, (0, 1, 3,2,4))
        Fixed_x=torch.permute(Fixed_x, (0, 1, 3,2,4))

        Moving_y[Moving_y>nlabels]=0
        Fixed_y[Fixed_y>nlabels]=0

        'Multi_channel Moving'
        Moving_y_mt = torch.zeros((Moving_y.size(0), nlabels,  *inshape))
        
        for organ_index in range(1,nlabels+1):
            temp_target = torch.zeros(Moving_y.size())
            temp_target[Moving_y == organ_index] = 1
            
            Moving_y_mt[:,organ_index-1,:,:,:]=torch.squeeze(temp_target)

        Fixed_y_mt = torch.zeros((Fixed_y.size(0), nlabels,  *inshape))
 
        for organ_index in range(1,nlabels+1):
            temp_target = torch.zeros(Fixed_y.size())
            temp_target[Fixed_y == organ_index] = 1
            
            Fixed_y_mt[:,organ_index-1,:,:,:]=torch.squeeze(temp_target)

        Moving_y= Moving_y_mt.cuda()   
        Fixed_y_mt=Fixed_y_mt.cuda()
        step_start_time = time.time()
        plt_simi_loss=0
        plt_smooth_loss=0
        plt_strcture_loss=0
        plt_seg_loss=0
        
        loss_list = []
        for iter_id in range(flow_num):

            if iter_id==0:
                states_h=None 
                states_c=None 
                Moving_x_def, Moving_y_def, pos_flow_cur,states_h,states_c = model(Moving_x,Moving_y,Fixed_x,states_h,states_c)
            else:
                Moving_x_def, Moving_y_def, pos_flow_cur,states_h,states_c = model(Moving_x_def,Moving_y_def,Fixed_x,states_h,states_c)
            
            reg_result_new = Moving_y_def

            ## Calculate loss
            Sim_loss=image_loss_func(Fixed_x,Moving_x_def)
            gradient_loss= grad_loss_func(pos_flow_cur)* smooth_w 
            plt_simi_loss=plt_simi_loss+Sim_loss
            plt_smooth_loss=plt_smooth_loss+gradient_loss


            # Landmark loss with smoothening to prevent excessive deformation of organs
            landmark_loss=structure_loss_cred(reg_result_new,Fixed_y_mt)
            plt_strcture_loss=plt_strcture_loss+landmark_loss
            optimizer.zero_grad()
            loss=Sim_loss + gradient_loss + args.seg_w*landmark_loss
            loss.backward()
            optimizer.step()

            Moving_x_def=Moving_x_def.detach()
            Moving_y_def=Moving_y_def.detach()

        loss= loss
        loss_info = 'loss: %.6f  (%s)' % (loss.item(),', '.join(loss_list))


        
        plt_strcture_loss=plt_strcture_loss/flow_num


        for seg_iter in range (0,flow_num):
            if seg_iter==0:
                h=None
                c=None
                y_pred,_,_,_,_ ,h,c,y_m_pred= model.forward_seg_training_all_enc_lstm(Moving_x,Fixed_x,Moving_y,h,c)
            else:
                y_pred,_,_,_,_,h,c,y_m_pred = model.forward_seg_training_all_enc_lstm(y_pred,Fixed_x,y_m_pred,h,c)        

            y_pred = y_pred.detach()
            y_m_pred = y_m_pred.detach()

            seg_in = torch.cat((Fixed_x,y_pred),1)
            seg_in = torch.cat((seg_in,y_m_pred),1)



            if seg_iter ==0:
                state_seg=None

            seg,h_seg,c_seg = model_seg3d(seg_in,state_seg)

            seg_result_new = seg
            seg_loss=seg_loss_cred(seg_result_new,Fixed_y)

            

            optimizer_seg.zero_grad()
            seg_loss.backward()
            optimizer_seg.step()
            
            state_seg=[h_seg.detach(),c_seg.detach()]
            plt_seg_loss=plt_seg_loss+seg_loss
            #Moving_x_tep=y_pred.detach()

        # print step info
        seg_loss_info = 'Seg loss: %.6f  (%s)' % (seg_loss.item(),', '.join(loss_list))
        Strcture_seg_loss_info = 'RegSeg loss: %.6f  (%s)' % (landmark_loss.item(),', '.join(loss_list))
        epoch_info = 'epoch: %04d' % (epoch + 1)
        #step_info = ('step: %d/%d' % (step + 1, args.steps_per_epoch)).ljust(14)
        time_info = 'time: %.2f sec' % (time.time() - step_start_time)

        #print('  '.join((epoch_info, time_info, loss_info,seg_loss_info)), flush=True)


        if total_steps%1==0:
            # Plotting graphs
            iter_count=iter_count+1
            plt_iternm.append(iter_count)
            plot_loss_value.append(plt_simi_loss.item())
            plot_loss_value1.append(plt_smooth_loss.item())
            plot_loss_value2.append(plt_strcture_loss.item()) 
            plot_loss_value3.append(plt_seg_loss.item())                

            ax.plot(plt_iternm,plot_loss_value,color='b',label='Simi_loss',linestyle='solid')
            ax.plot(plt_iternm,plot_loss_value1,color='k',label='Smooth_loss',linestyle='dashed')
            ax.plot(plt_iternm,plot_loss_value2,color='r',label='Strcture_loss',linestyle='solid') 
            #ax.plot(plt_iternm,plot_loss_value3,color='magenta',label='Seg_loss',linestyle='dotted') 

                        

            plt.xlabel('iteration')
            plt.ylabel('errors/accuracys')
            
            if iter_count==1:
                plt.legend()
            
            plt_name=sv_folder+'error_plot_sim_smooth.png'        
            plt.savefig(plt_name,bbox_inches='tight')     
    


            print('  '.join((epoch_info, time_info, loss_info,seg_loss_info,Strcture_seg_loss_info)), flush=True)

            if (train_sv_flag == 0):
                train_sv_flag = 1
                ### Saving one training image
                y_pred_show=torch.squeeze(Moving_x_def,1)
                Moving_x_show=torch.squeeze(Moving_x,1)
                Fixed_x_show=torch.squeeze(Fixed_x,1)
                seg_show=torch.argmax(seg, dim=1)
                seg_show=torch.squeeze(seg_show,0)
                Fixed_y_show=torch.squeeze(Fixed_y,1)

                'save the images'
                y_pred_show=y_pred_show.data.cpu().numpy()
                y_pred_show=y_pred_show[0]
                y_pred_show  = np.transpose(y_pred_show, (1, 0, 2))

                y_pred_show = nib.Nifti1Image(y_pred_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'deformed_Moving.nii'
                nib.save(y_pred_show, pred_sv_name)   
                
                Moving_x_show=Moving_x_show.data.cpu().numpy()
                Moving_x_show=Moving_x_show[0]
                Moving_x_show  = np.transpose(Moving_x_show, (1, 0, 2))

                Moving_x_show = nib.Nifti1Image(Moving_x_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_Moving.nii'
                nib.save(Moving_x_show, pred_sv_name)  

                Fixed_x_show=Fixed_x_show.data.cpu().numpy()
                Fixed_x_show=Fixed_x_show[0]
                Fixed_x_show  = np.transpose(Fixed_x_show, (1, 0, 2))

                Fixed_x_show = nib.Nifti1Image(Fixed_x_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_Fixed.nii'
                nib.save(Fixed_x_show, pred_sv_name)  

                Fixed_y_show=Fixed_y_show.data.cpu().numpy()
                Fixed_y_show=Fixed_y_show[0]
                Fixed_y_show  = np.transpose(Fixed_y_show, (1, 0, 2))

                Fixed_y_show = nib.Nifti1Image(Fixed_y_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_Fixed_msk.nii'
                nib.save(Fixed_y_show, pred_sv_name)  

    if epoch%1==0:
        with torch.no_grad(): # no grade calculation 

            best_reg_dsc = 0
            best_seg_dsc = 0
            print('Validation')
            for i_iter_val, item_val in enumerate(val_loader):    
                # print('VAL')
                

                Fixed_val_img=item_val['fixed_img'].float().cuda()
                Fixed_val_msk=item_val['fixed_msk'].float().cuda()
                Moving_img=item_val['move_img'].float().cuda()
                Moving_val_msk=item_val['move_msk'].float().cuda()
                source_name=item_val['fixed_img'].meta['filename_or_obj'][0].split('/')[-1]
                target_name =item_val['move_img'].meta['filename_or_obj'][0].split('/')[-1]


                Moving_img=torch.permute(Moving_img, (0, 1, 3,2,4))
                Moving_val_msk=torch.permute(Moving_val_msk, (0, 1, 3,2,4))
                Fixed_val_img=torch.permute(Fixed_val_img, (0, 1, 3,2,4))
                Fixed_val_msk=torch.permute(Fixed_val_msk, (0, 1, 3,2,4))

                Moving_val_msk[Moving_val_msk>nlabels]=0
                Fixed_val_msk[Fixed_val_msk>nlabels]=0


                'Multi_channel moving'
                Moving_val_msk_mt = torch.zeros((Moving_val_msk.size(0), nlabels,  *inshape))
                
                for organ_index in range(1,1+nlabels):
                    temp_target = torch.zeros(Moving_val_msk.size())
                    temp_target[Moving_val_msk == organ_index] = 1
                    
                    Moving_val_msk_mt[:,organ_index-1,:,:,:]=torch.squeeze(temp_target)

                Moving_val_msk= Moving_val_msk_mt.cuda()   
                
                # feed the data in
                for seg_iter_val in range (0,flow_num+1):
                    if seg_iter_val==0:
                        h=None
                        c=None
                        y_pred_val,dvf_flow,_,_,_ ,h,c,y_m_pred_val= model.forward_seg_training_all_enc_lstm(Moving_img,Fixed_val_img,Moving_val_msk,h,c)
                    else:
                        y_pred_val,dvf_flow,_,_,_,h,c,y_m_pred_val = model.forward_seg_training_all_enc_lstm(y_pred_val,Fixed_val_img,y_m_pred_val,h,c) 
                
                    if seg_iter_val ==0:
                        state_seg=None
                    seg_in_val=torch.cat((Fixed_val_img,y_pred_val),1)
                    seg_in_val=torch.cat((seg_in_val,y_m_pred_val),1)
                    seg_result,h_seg,c_seg=model_seg3d(seg_in_val,state_seg)
                    state_seg=[h_seg,c_seg]

                seg_result=torch.argmax(seg_result, dim=1)
                
                reg_result=torch.zeros(1, *inshape)
                for index in range(0,nlabels):
                    reg_result[y_m_pred_val[:,index,:,:,:]>0.5]=index+1

                info = {'Source_Name':source_name, 'Target_Name':target_name}

                spacing = (1.0,1.0,1.0) # Not actual spacing, needs to be fixed
                eval.calculate_results(info, spacing, Fixed_val_msk, Moving_val_msk, reg_result, seg_result, dvf_flow)
        
        info = eval.average_results()
        seg_dice = float(info['Seg_Avg_Dice'][:nlabels])
        reg_dice = float(info['Reg_Avg_Dice'][:nlabels])

        if seg_dice> best_seg_dsc:
            model_sv_path=sv_folder+'sv_reg_model_seg.pt'
            model.save(model_sv_path)

            best_seg_dsc = seg_dice
            model_seg_sv_path=sv_folder+'sv_seg_model_seg.pt'
            torch.save(model_seg3d.state_dict(), model_seg_sv_path)

        if reg_dice > best_reg_dsc:

            best_reg_dsc = reg_dice
            model_sv_path=sv_folder+'sv_reg_model_reg.pt'
            model.save(model_sv_path)

            model_seg_sv_path=sv_folder+'sv_reg_model_seg.pt'
            torch.save(model_seg3d.state_dict(), model_seg_sv_path)


    model_sv_path=sv_folder+'final_sv_reg_model_seg.pt'
    model.save(model_sv_path)

    model_seg_sv_path=sv_folder+'final_sv_seg_model_seg.pt'
    torch.save(model_seg3d.state_dict(), model_seg_sv_path)


        
    model_sv_path=sv_folder+'final_sv_reg_model_reg.pt'
    model.save(model_sv_path)
        
            

                        







