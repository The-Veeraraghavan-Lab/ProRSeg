
import os
from monai import transforms, data
from monai.data import load_decathlon_datalist

from monai.transforms import OneOf,RandCoarseDropoutd

def get_loader_Nishant(args):
    data_dir = args.datadir
    json_list = args.json
    datalist_json = os.path.join(data_dir, json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=0),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=1),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=2),
            transforms.RandAffined(keys=["move_img","move_msk"],
                                   prob=0.4,
                                   rotate_range=0.0872665,
                                   translate_range=5,
                                   mode = ("bilinear", "nearest")
                                ),
                                
            #transforms.RandRotate90d(
            #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
            #    prob=0.3,
            #    max_k=3,
            #),
            #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
            #                               factors=0.3,
            #                               prob=0.3),
            #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
            #                               offsets=0.3,
            #                               prob=0.3),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    
    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "train",
                                        base_dir=data_dir)
    #datalist=datalist[0:10]
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
                                        "validation",
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
    #loader = [train_loader, val_loader]

    return train_loader,val_loader

def get_loader_MRI_pancreas_Nishant(args):
    data_dir = '/lab/deasylab1/Jue/MRI_train_data/data_extct_code/extracted_all_data_histo_matched_128_192_128/'
    json_list='data_json_train_val.json'
    datalist_json = os.path.join(data_dir, json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=0),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=1),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=2),
            transforms.RandAffined(keys=["move_img","move_msk"],
                                   prob=0.4,
                                   rotate_range=0.0872665,
                                   translate_range=5,
                                   mode = ("bilinear", "nearest")
                                ),
                                
            #transforms.RandRotate90d(
            #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
            #    prob=0.3,
            #    max_k=3,
            #),
            #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
            #                               factors=0.3,
            #                               prob=0.3),
            #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
            #                               offsets=0.3,
            #                               prob=0.3),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    
    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "MRI_pancreas_train",
                                        base_dir=data_dir)
    #datalist=datalist[0:10]
    train_ds = data.Dataset(data=datalist, transform=train_transform)

    if 2>1:
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
                                        "MRI_pancreas_val",
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
    #loader = [train_loader, val_loader]

    return train_loader,val_loader


def get_loader_MRI_pancreas_Nishant_K_fold(args):
    data_dir = '/lab/deasylab1/Jue/MRI_train_data/data_extct_code/extracted_all_data_histo_matched_128_192_128/'
    json_list='train_val_fold_1.json'
    
    datalist_json = os.path.join(data_dir, json_list)
    if args.affine:
        print("AFFINE")
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
                
                transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                                a_min=0,
                                                a_max=2000,
                                                b_min=0,
                                                b_max=1,
                                                clip=True),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=0),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=1),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=2),
                transforms.RandAffined(keys=["move_img","move_msk"],
                                    prob=0.4,
                                    rotate_range=0.174533,
                                    translate_range=5,
                                    mode = ("bilinear", "nearest")
                                    ),
                                    
                #transforms.RandRotate90d(
                #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                #    prob=0.3,
                #    max_k=3,
                #),
                #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
                #                               factors=0.3,
                #                               prob=0.3),
                #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
                #                               offsets=0.3,
                #                               prob=0.3),
                transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
                
                transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                                a_min=0,
                                                a_max=2000,
                                                b_min=0,
                                                b_max=1,
                                                clip=True),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=0),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=1),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=2),
                # transforms.RandAffined(keys=["move_img","move_msk"],
                #                     prob=0.4,
                #                     rotate_range=0.174533,
                #                     translate_range=5,
                #                     mode = ("bilinear", "nearest")
                #                     ),
                                    
                #transforms.RandRotate90d(
                #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                #    prob=0.3,
                #    max_k=3,
                #),
                #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
                #                               factors=0.3,
                #                               prob=0.3),
                #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
                #                               offsets=0.3,
                #                               prob=0.3),
                transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    
    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "MRI_pancreas_train",
                                        base_dir=data_dir)
    #datalist=datalist[0:10]
    train_ds = data.Dataset(data=datalist, transform=train_transform)

    if 2>1:
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
                                        "MRI_pancreas_val",
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
    #loader = [train_loader, val_loader]

    return train_loader,val_loader

def get_loader_MRI_pancreas_Nishant_K_fold_val(args):
    data_dir = '/lab/deasylab1/Jue/MRI_train_data/data_extct_code/extracted_all_data_histo_matched_128_192_128/'
    # json_list='train_val_fold_1.json'
    json_list='data_json_train_val_10.json'
    datalist_json = os.path.join(data_dir, json_list)
    if args.affine:
        print("AFFINE")
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
                
                transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                                a_min=0,
                                                a_max=2000,
                                                b_min=0,
                                                b_max=1,
                                                clip=True),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=0),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=1),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=2),
                transforms.RandAffined(keys=["move_img","move_msk"],
                                    prob=0.4,
                                    rotate_range=0.174533,
                                    translate_range=5,
                                    mode = ("bilinear", "nearest")
                                    ),
                                    
                #transforms.RandRotate90d(
                #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                #    prob=0.3,
                #    max_k=3,
                #),
                #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
                #                               factors=0.3,
                #                               prob=0.3),
                #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
                #                               offsets=0.3,
                #                               prob=0.3),
                transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
                #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
                
                transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                                a_min=0,
                                                a_max=2000,
                                                b_min=0,
                                                b_max=1,
                                                clip=True),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=0),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=1),

                transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                    prob=0.3,
                                    spatial_axis=2),
                # transforms.RandAffined(keys=["move_img","move_msk"],
                #                     prob=0.4,
                #                     rotate_range=0.174533,
                #                     translate_range=5,
                #                     mode = ("bilinear", "nearest")
                #                     ),
                                    
                #transforms.RandRotate90d(
                #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                #    prob=0.3,
                #    max_k=3,
                #),
                #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
                #                               factors=0.3,
                #                               prob=0.3),
                #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
                #                               offsets=0.3,
                #                               prob=0.3),
                transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    
    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "MRI_pancreas_train",
                                        base_dir=data_dir)
    #datalist=datalist[0:10]
    train_ds = data.Dataset(data=datalist, transform=train_transform)

    if 2>1:
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
                                        "MRI_pancreas_val",
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
    #loader = [train_loader, val_loader]

    return train_loader,val_loader


def get_loader_MRI_pancreas(args):
    data_dir = '/lab/deasylab1/Jue/MRI_train_data/data_extct_code/extracted_all_data_histo_matched_128_192_128/'
    json_list='data_json_train_val.json'
    datalist_json = os.path.join(data_dir, json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img","fix_img","move_msk","fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=0),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=1),

            transforms.RandFlipd(keys=["move_img", "fix_img", "move_msk", "fix_msk"],
                                 prob=0.3,
                                 spatial_axis=2),

            #transforms.RandRotate90d(
            #    keys=["move_img", "fix_img", "move_msk", "fix_msk"],
            #    prob=0.3,
            #    max_k=3,
            #),
            #transforms.RandScaleIntensityd(keys=["move_img","fix_img"],
            #                               factors=0.3,
            #                               prob=0.3),
            #transforms.RandShiftIntensityd(keys=["move_img","fix_img"],
            #                               offsets=0.3,
            #                               prob=0.3),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            #transforms.Orientationd(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=0,
                                            a_max=2000,
                                            b_min=0,
                                            b_max=1,
                                            clip=True),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    
    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "MRI_pancreas_train",
                                        base_dir=data_dir)
    
    train_ds = data.Dataset(data=datalist, transform=train_transform)
    if 3>2:
        train_ds = data.CacheDataset(
            data=datalist,
            transform=train_transform,
            cache_num= 800,#1200,#400,#200,#args.cache_num,# 250,
            cache_rate=1.0,
            num_workers=args.workers,
        )
    train_sampler = None
    train_loader = data.DataLoader(train_ds,
                                    batch_size=3,#args.batch_size,
                                    #batch_size=args.batch_size,
                                    shuffle=(train_sampler is None),
                                    num_workers=args.workers,
                                    sampler=train_sampler,
                                    pin_memory=True,
                                    persistent_workers=True)
    val_files = load_decathlon_datalist(datalist_json,
                                        True,
                                        "MRI_pancreas_val",
                                        base_dir=data_dir)
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler =  None
    val_loader = data.DataLoader(val_ds,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    sampler=val_sampler,
                                    pin_memory=True,
                                    persistent_workers=True)
    #loader = [train_loader, val_loader]

    return train_loader,val_loader
