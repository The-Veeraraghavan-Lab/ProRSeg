#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import torch.nn as nn
import torch
import matplotlib
matplotlib.use('Agg')

from monai_data_loader_mri_pancreas import get_loader_Nishant
import matplotlib.pyplot as plt
from torch.autograd import Variable

fig = plt.figure()
ax = fig.add_subplot(211)
import torch.nn.functional as F

plot_loss_value=[]
plot_loss_value1=[]
plot_loss_value2=[]


import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import torch.nn.functional as F
from torch.autograd import Variable
from monai.losses import DiceCELoss
import math 
import json
from evaluation import Eval

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True


def save_args(args, filename='args.json'):
    with open(filename, 'w') as file:
        json.dump(vars(args), file, indent=4)

'''
python train.py --smooth 150 --seg_w 3 --image_loss ncc --win_sz 3 --affine --svdir model_save_dirs/train_ncc_win_3 --cache

python train.py --smooth 150 --seg_w 3 --image_loss ncc --win_sz 3 --affine --svdir model_save_dirs/test --cache

'''
# parse the commandline
parser = argparse.ArgumentParser()
fig = plt.figure()
ax = fig.add_subplot(211)
# data organization parameters
parser.add_argument('--datadir', default='/lab/deasylab1/Nishant/ProRSeg/Dataset/Old_17_cases/histogram_matched_data', help='base data directory')
parser.add_argument('--json', default='17_cases.json', help='base data directory')
parser.add_argument('--model_dir', default='models', help='model output directory (default: models)')
parser.add_argument('--cache',  action='store_true', help='Cache training dataset for faster training')

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
model_seg3d = vxm.networks.UNet3D_Seg_LSTM(in_channels=1+1+4,out_channels=4+1,final_sigmoid=False)

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




structure_loss_cred = vxm.losses.DiceLoss_test_use()

train_loader_a,val_loader=get_loader_Nishant(args)

total_steps=0       

plt_iternm=[]
flow_num=args.flownum
iter_count=0

flow_ini=torch.zeros(1, 3, *inshape).cuda()
range_flow=1


class_list = ['Liver', 'Lg_Bowel','Sm_Bowel','Duo_Stomach']
eval = Eval(train_tep_sv, class_list)

for epoch in range(args.initial_epoch, args.epochs):
    eval.update_epoch((epoch+1))
    train_sv_flag = 0
    print ('running in epoch',epoch)
    if epoch >1 and epoch % 5 ==0:
        model_sv_path=sv_folder+str(epoch)+'_reg_model.pt'
        model.save(model_sv_path)

        model_seg_sv_path=sv_folder+str(epoch)+'_seg_model.pt'
        torch.save(model_seg3d.state_dict(), model_seg_sv_path)


    

           


    for i_iter, item_ in enumerate(train_loader_a): 
        #print (item_.keys())
        #print  ('cbct size ',cbct_x.size())
        total_steps=total_steps+1
        cbct_x=item_['fix_img'].float().cuda()
        cbct_y=item_['fix_msk'].float().cuda()
        planct_x=item_['move_img'].float().cuda()
        planct_y=item_['move_msk'].float().cuda()



        planct_x=torch.permute(planct_x, (0, 1, 2,4,3))
        planct_y=torch.permute(planct_y, (0, 1, 2,4,3))
        cbct_y=torch.permute(cbct_y, (0, 1, 2,4,3))
        cbct_x=torch.permute(cbct_x, (0, 1, 2,4,3))

        #print ('info: planct_x size ',planct_x.size())
        planct_x=torch.permute(planct_x, (0, 1, 4,3,2))
        planct_y=torch.permute(planct_y, (0, 1,  4,3,2))
        cbct_y=torch.permute(cbct_y, (0, 1,  4,3,2))
        cbct_x=torch.permute(cbct_x, (0, 1, 4,3,2))


        planct_x=torch.flip(planct_x, [2, 3])
        planct_y=torch.flip(planct_y, [2, 3])
        cbct_y=torch.flip(cbct_y, [2, 3])
        cbct_x=torch.flip(cbct_x, [2, 3])


        planct_y[planct_y>4]=0
        cbct_y[cbct_y>4]=0

        'Multi_channel PlanCT'
        PlanCT_y_mt = torch.zeros((planct_y.size(0), 4,  *inshape))
        
        for organ_index in range(1,5):
            temp_target = torch.zeros(planct_y.size())
            temp_target[planct_y == organ_index] = 1
            
            PlanCT_y_mt[:,organ_index-1,:,:,:]=torch.squeeze(temp_target)

        CBCT_y_mt = torch.zeros((cbct_y.size(0), 4,  *inshape))
 
        for organ_index in range(1,5):
            temp_target = torch.zeros(cbct_y.size())
            temp_target[cbct_y == organ_index] = 1
            
            CBCT_y_mt[:,organ_index-1,:,:,:]=torch.squeeze(temp_target)

        planct_y= PlanCT_y_mt.cuda()   
        CBCT_y_mt=CBCT_y_mt.cuda()
        step_start_time = time.time()
        plt_simi_loss=0
        plt_smooth_loss=0
        plt_strcture_loss=0
        
        loss_list = []
        for iter_id in range(flow_num):

            if iter_id==0:
                states_h=None 
                states_c=None 
                planct_x_def, planct_y_def, pos_flow_cur,states_h,states_c = model(planct_x,planct_y,cbct_x,states_h,states_c)
            else:
                planct_x_def, planct_y_def, pos_flow_cur,states_h,states_c = model(planct_x_def,planct_y_def,cbct_x,states_h,states_c)

            # reg_result_new = torch.squeeze(planct_y_def.clone()).data.cpu().numpy()

            ### Code to deal with incomplete segmentations (2 cm ring around the tumor)
            # for label_num in range(planct_y_def.shape[1]):
            #     if (np.sum(torch.squeeze(planct_y_def).data.cpu().numpy()[label_num:,:,105]>0)==0):
            #         flag = 0
            #         for i in range(60, planct_y_def.shape[-1]):
            #             slice = torch.squeeze(CBCT_y_mt).data.cpu().numpy()[label_num,:,:,i]
                        
            #             if flag == 0:
            #                 if np.sum(slice>0) < 100:
            #                     flag = 1
            #             else:
            #                 # seg_result_new[:,:,i] = 0
            #                 reg_result_new[label_num, :,:,i] = 0

            # reg_result_new = np.expand_dims(reg_result_new, axis=0)
            # reg_result_new = torch.tensor(reg_result_new).cuda()
            
            reg_result_new = planct_y_def

            ## Calculate loss
            Sim_loss=image_loss_func(cbct_x,planct_x_def)
            gradient_loss= grad_loss_func(pos_flow_cur)* smooth_w 
            plt_simi_loss=plt_simi_loss+Sim_loss
            plt_smooth_loss=plt_smooth_loss+gradient_loss

            # seg_in_r=torch.cat((cbct_x,planct_x_def),1)
            # seg_in_r=torch.cat((seg_in_r,planct_y_def),1)

            # if iter_id ==0:
            #     state_seg_r=None
            
            # Landmark loss with smoothening to prevent excessive deformation of organs
            landmark_loss=structure_loss_cred(reg_result_new,CBCT_y_mt)
            plt_strcture_loss=plt_strcture_loss+landmark_loss
            optimizer.zero_grad()
            loss=Sim_loss + gradient_loss + args.seg_w*landmark_loss
            loss.backward()
            optimizer.step()

            planct_x_def=planct_x_def.detach()
            planct_y_def=planct_y_def.detach()

        loss= loss
        loss_info = 'loss: %.6f  (%s)' % (loss.item(),', '.join(loss_list))


        
        plt_strcture_loss=plt_strcture_loss/flow_num


        for seg_iter in range (0,flow_num+1):
            #print (seg_iter)
            if seg_iter==0:
                h=None
                c=None
                y_pred,_,_,_,_ ,h,c,y_m_pred= model.forward_seg_training_all_enc_lstm(planct_x,cbct_x,planct_y,h,c)
            else:
                y_pred,_,_,_,_,h,c,y_m_pred = model.forward_seg_training_all_enc_lstm(y_pred,cbct_x,y_m_pred,h,c)        

            y_pre = y_pred.detach()
            y_m_pred = y_m_pred.detach()

            seg_in = torch.cat((cbct_x,y_pred),1)
            seg_in = torch.cat((seg_in,y_m_pred),1)



            if seg_iter ==0:
                state_seg=None

            seg,h_seg,c_seg = model_seg3d(seg_in,state_seg)


            ### Code to deal with incomplete segmentations (2 cm ring around the tumor)
            # seg_result_new = torch.squeeze(seg.clone()).data.cpu().numpy()

            # for label_num in range(seg.shape[1]):
            #     if (np.sum(torch.squeeze(cbct_y).data.cpu().numpy()[:,:,105]>0)==0):
            #         flag = 0
            #         for i in range(60, cbct_y.shape[-1]):
            #             slice = torch.squeeze(cbct_y).data.cpu().numpy()[:,:,i]
                        
            #             if flag == 0:
            #                 if np.sum(slice>0) < 300:
            #                     flag = 1
            #             else:
            #                 # seg_result_new[:,:,i] = 0
            #                 seg_result_new[label_num, :,:,i] = 0
            # seg_result_new = np.expand_dims(seg_result_new, axis=0)
            # seg_result_new = torch.tensor(seg_result_new).cuda()

            # print ('seg size ',seg.size())
            # print ('cbct_y size ',cbct_y.size())
            seg_result_new = seg
            seg_loss=seg_loss_cred(seg_result_new,cbct_y)

            optimizer_seg.zero_grad()
            seg_loss.backward()
            optimizer_seg.step()
            state_seg=[h_seg.detach(),c_seg.detach()]
            #planct_x_tep=y_pred.detach()

        # print step info
        seg_loss_info = 'Seg loss: %.6f  (%s)' % (seg_loss.item(),', '.join(loss_list))
        Strcture_seg_loss_info = 'Seg loss: %.6f  (%s)' % (landmark_loss.item(),', '.join(loss_list))
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
                        

            ax.plot(plt_iternm,plot_loss_value,color='r',label='Simi_loss',linestyle='solid')
            ax.plot(plt_iternm,plot_loss_value1,color='b',label='Smooth_loss',linestyle='dashed')
            ax.plot(plt_iternm,plot_loss_value2,color='k',label='Strcture_loss',linestyle='solid') 
                        #ax.plot(plt_iternm,plot_loss_value2,color='b',label='Seg_loss',linestyle='solid')

                        

            plt.xlabel('interation times')
            plt.ylabel('errors/accuracys')

            plt_name=sv_folder+'error_plot_sim_smooth.png'        
            plt.savefig(plt_name,bbox_inches='tight')     
    


            print('  '.join((epoch_info, time_info, loss_info,seg_loss_info,Strcture_seg_loss_info)), flush=True)

            if train_sv_flag == 0:
                train_sv_flag = 1
                ### Saving one training image
                y_pred_show=torch.squeeze(y_pred)
                planct_x_show=torch.squeeze(planct_x)
                cbct_x_show=torch.squeeze(cbct_x)
                y_m_pred_show=torch.squeeze(y_m_pred)
                seg_show=torch.argmax(seg, dim=1)
                seg_show=torch.squeeze(seg_show)
                cbct_y_show=torch.squeeze(cbct_y)

                'save the images'
                y_pred_show=y_pred_show.data.cpu().numpy()
                y_pred_show=y_pred_show[0]
                y_pred_show = np.squeeze(y_pred_show)
                y_pred_show = np.flip(y_pred_show, axis=2)
                y_pred_show  = np.transpose(y_pred_show, (1, 0, 2))

                y_pred_show = nib.Nifti1Image(y_pred_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'deformed_planCT.nii'
                nib.save(y_pred_show, pred_sv_name)   

                y_m_pred_show=y_m_pred_show.data.cpu().numpy()
                y_m_pred_show=y_m_pred_show[0]
                y_m_pred_show=np.squeeze(y_m_pred_show)
                y_m_pred_show = np.flip(y_m_pred_show, axis=2)
                y_m_pred_show  = np.transpose(y_m_pred_show, (1, 0, 2))

                y_m_pred_show = nib.Nifti1Image(y_m_pred_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'deformed_planCT_msk.nii'
                nib.save(y_m_pred_show, pred_sv_name)   
                
                planct_x_show=planct_x_show.data.cpu().numpy()
                planct_x_show=planct_x_show[0]
                planct_x_show=np.squeeze(planct_x_show)
                planct_x_show = np.flip(planct_x_show, axis=2)
                planct_x_show  = np.transpose(planct_x_show, (1, 0, 2))

                planct_x_show = nib.Nifti1Image(planct_x_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_planCT.nii'
                nib.save(planct_x_show, pred_sv_name)  

                cbct_x_show=cbct_x_show.data.cpu().numpy()
                cbct_x_show=cbct_x_show[0]
                cbct_x_show=np.squeeze(cbct_x_show)
                cbct_x_show = np.flip(cbct_x_show, axis=2)
                cbct_x_show  = np.transpose(cbct_x_show, (1, 0, 2))

                cbct_x_show = nib.Nifti1Image(cbct_x_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_CBCT.nii'
                nib.save(cbct_x_show, pred_sv_name)  

                cbct_y_show=cbct_y_show.data.cpu().numpy()
                cbct_y_show=cbct_y_show[0]
                cbct_y_show=np.squeeze(cbct_y_show)
                cbct_y_show = np.flip(cbct_y_show, axis=2)
                cbct_y_show  = np.transpose(cbct_y_show, (1, 0, 2))

                cbct_y_show = nib.Nifti1Image(cbct_y_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_CBCT_msk.nii'
                nib.save(cbct_y_show, pred_sv_name)  


                seg_show=seg_show.data.cpu().float().numpy()
                seg_show=seg_show[0]
                seg_show=np.squeeze(seg_show)
                seg_show = np.flip(seg_show, axis=2)
                seg_show  = np.transpose(seg_show, (1, 0, 2))

                seg_show = nib.Nifti1Image(seg_show,np.eye(4))    
                pred_sv_name=train_tep_sv+'ori_CBCT_seg.nii'
                nib.save(seg_show, pred_sv_name)  


            

    if epoch%30==0:
        with torch.no_grad(): # no grade calculation 

            best_reg_dsc = 0
            best_seg_dsc = 0
            print('Validation')
            for i_iter_val, item_val in enumerate(val_loader):    
                # print('VAL')
                

                cbct_val_img=item_val['fix_img'].float().cuda()
                cbct_val_msk=item_val['fix_msk'].float().cuda()
                plan_ct_img=item_val['move_img'].float().cuda()
                planct_val_msk=item_val['move_msk'].float().cuda()
                source_name = item_val['move_img_meta_dict']['filename_or_obj'][0].split('/')[-1]
                target_name = item_val['fix_img_meta_dict']['filename_or_obj'][0].split('/')[-1]

                plan_ct_img=torch.permute(plan_ct_img, (0, 1, 2,4,3))
                planct_val_msk=torch.permute(planct_val_msk, (0, 1, 2,4,3))
                cbct_val_img=torch.permute(cbct_val_img, (0, 1, 2,4,3))
                cbct_val_msk=torch.permute(cbct_val_msk, (0, 1, 2,4,3))


                plan_ct_img=torch.permute(plan_ct_img, (0, 1, 4,3,2))
                planct_val_msk=torch.permute(planct_val_msk, (0, 1,4,3,2))
                cbct_val_img=torch.permute(cbct_val_img, (0, 1, 4,3,2))
                cbct_val_msk=torch.permute(cbct_val_msk, (0, 1,4,3,2))


                plan_ct_img=torch.flip(plan_ct_img, [2, 3])
                planct_val_msk=torch.flip(planct_val_msk, [2, 3])
                cbct_val_img=torch.flip(cbct_val_img, [2, 3])
                cbct_val_msk=torch.flip(cbct_val_msk, [2, 3])


                planct_val_msk[planct_val_msk>4]=0
                cbct_val_msk[cbct_val_msk>4]=0


                'Multi_channel PlanCT'
                PlanCT_val_msk_mt = torch.zeros((planct_val_msk.size(0), 4,  *inshape))
                
                for organ_index in range(1,5):
                    temp_target = torch.zeros(planct_val_msk.size())
                    temp_target[planct_val_msk == organ_index] = 1
                    
                    PlanCT_val_msk_mt[:,organ_index-1,:,:,:]=torch.squeeze(temp_target)

                planct_val_msk= PlanCT_val_msk_mt.cuda()   
                
                # feed the data in
                for seg_iter_val in range (0,flow_num+1):
                    if seg_iter_val==0:
                        h=None
                        c=None
                        y_pred_val,dvf_flow,_,_,_ ,h,c,y_m_pred_val= model.forward_seg_training_all_enc_lstm(plan_ct_img,cbct_val_img,planct_val_msk,h,c)
                    else:
                        y_pred_val,dvf_flow,_,_,_,h,c,y_m_pred_val = model.forward_seg_training_all_enc_lstm(y_pred_val,cbct_val_img,y_m_pred_val,h,c) 
                
                    if seg_iter_val ==0:
                        state_seg=None
                    seg_in_val=torch.cat((cbct_val_img,y_pred_val),1)
                    seg_in_val=torch.cat((seg_in_val,y_m_pred_val),1)
                    seg_result,h_seg,c_seg=model_seg3d(seg_in_val,state_seg)
                    state_seg=[h_seg,c_seg]

                seg_result=torch.argmax(seg_result, dim=1)
                

                #print (planct_val_msk.size())
                #print (y_m_pred_val.size())
                reg_result=torch.zeros(1, *inshape)
                reg_result[y_m_pred_val[:,0,:,:,:]>0.5]=1
                reg_result[y_m_pred_val[:,1,:,:,:]>0.5]=2
                reg_result[y_m_pred_val[:,2,:,:,:]>0.5]=3
                reg_result[y_m_pred_val[:,3,:,:,:]>0.5]=4


                #y_m_pred_val=torch.argmax(y_m_pred_val, dim=1)
                
                # gt=cbct_val_msk.float().cpu().numpy()
                # print(gt.shape)

                # seg_result=seg_result.float().cpu().numpy()
                # reg_result=reg_result.float().cpu().numpy()

                # print(seg_result.shape)
                # seg_result_new = np.squeeze(seg_result.copy())
                # reg_result_new = np.squeeze(reg_result.copy())

                ### Code to deal with incomplete segmentations (2 cm ring around the tumor)
                # if (np.sum(torch.squeeze(cbct_val_msk).data.cpu().numpy()[:,:,105]>0)==0):
                #     flag = 0
                #     for i in range(60, planct_y_def.shape[-1]):
                #         slice = torch.squeeze(cbct_val_msk).data.cpu().numpy()[:,:,i]
                        
                #         if flag == 0:
                #             if np.sum(slice>0) < 300:
                #                 flag = 1
                #         else:
                #             seg_result_new[:,:,i] = 0
                #             reg_result_new[:,:,i] = 0

                #print (dsc_reg_4)
                info = {'Source_Name':source_name, 'Target_Name':target_name}

                spacing = (1.0,1.0,1.0) # Not actual spacing, needs to be fixed
                eval.calculate_results(info, spacing, cbct_val_msk, planct_val_msk, reg_result, seg_result, dvf_flow)
        
        info = eval.average_results()
        seg_dice = float(info['Seg_Avg_Dice'][:4])
        reg_dice = float(info['Reg_Avg_Dice'][:4])

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
        
            

                        



