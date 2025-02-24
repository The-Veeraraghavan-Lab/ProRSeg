
import os
from monai import transforms, data
from monai.data import load_decathlon_datalist


def get_dataloader(args):
    data_dir = args.datadir
    json_list = args.json
    datalist_json = os.path.join(data_dir, json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
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
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            transforms.AddChanneld(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
            
            transforms.ScaleIntensityRanged(keys=["move_img","fix_img"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.ToTensord(keys=["move_img", "fix_img", "move_msk", "fix_msk"]),
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
            cache_num= args.cache_num,
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

    return train_loader,val_loader
