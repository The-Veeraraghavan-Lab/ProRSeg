
import os
from monai import transforms, data
from monai.data import load_decathlon_datalist


def get_dataloader(args):
      data_dir = args.datadir
      json_list = args.json
      datalist_json = os.path.join(data_dir, json_list)
      train_transform = transforms.Compose(
          [
              transforms.LoadImaged(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
              transforms.EnsureChannelFirstd(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
              #transforms.Orientationd(keys=["move_img","fixed_img","move_msk","fixed_msk"]),
              
              transforms.ScaleIntensityRanged(keys=["move_img","fixed_img"],
                                              a_min=0,
                                              a_max=2000,
                                              b_min=0,
                                              b_max=1,
                                              clip=True),
    
              transforms.RandFlipd(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"],
                                   prob=0.3,
                                   spatial_axis=0),
    
              transforms.RandFlipd(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"],
                                   prob=0.3,
                                   spatial_axis=1),
    
              transforms.RandFlipd(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"],
                                   prob=0.3,
                                   spatial_axis=2),
              transforms.RandAffined(keys=["move_img","move_msk"],
                                     prob=0.4,
                                     rotate_range=0.0872665,
                                     translate_range=5,
                                     mode = ("bilinear", "nearest")
                                  ),
              transforms.ToTensord(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
          ]
      )
    
      val_transform = transforms.Compose(
          [
              transforms.LoadImaged(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
              transforms.EnsureChannelFirstd(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
              #transforms.Orientationd(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
              
              transforms.ScaleIntensityRanged(keys=["move_img","fixed_img"],
                                              a_min=0,
                                              a_max=2000,
                                              b_min=0,
                                              b_max=1,
                                              clip=True),
              transforms.ToTensord(keys=["move_img", "fixed_img", "move_msk", "fixed_msk"]),
          ]
      )
    
      
      datalist = load_decathlon_datalist(datalist_json,
                                          True,
                                          "train",
                                          base_dir=data_dir)
      
      train_ds = data.Dataset(data=datalist, transform=train_transform)
    
      if args.cache:
          train_ds = data.CacheDataset(
              data=datalist,
              transform=train_transform,
              cache_num= 200,#400,#1200,#400,#200,#args.cache_num,# 250,
              cache_rate=1.0,
              num_workers=args.workers,
          )
      train_sampler = None
    
      persns_flag= (args.workers>0)
    
      train_loader = data.DataLoader(train_ds,
                                      batch_size=args.batch_size,
                                      shuffle=(train_sampler is None),
                                      num_workers=args.workers,
                                      sampler=train_sampler,
                                      pin_memory=True,
                                      persistent_workers=persns_flag,
                                      drop_last=True)
      val_files = load_decathlon_datalist(datalist_json,
                                          True,
                                          "val",
                                          base_dir=data_dir)
      val_ds = data.Dataset(data=val_files, transform=val_transform)
      val_sampler =  None
      val_loader = data.DataLoader(val_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      sampler=val_sampler,
                                      pin_memory=True,
                                      persistent_workers=persns_flag)

      return train_loader,val_loader
