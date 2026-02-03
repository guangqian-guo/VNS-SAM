# train set
# COD
dataset_cod = {"name": "COD",
                "im_dir": "./data/COD/TrainDataset/Imgs",
                "gt_dir": "./data/COD/TrainDataset/GT",
                "edge_dir": "./data/COD/TrainDataset/Edge",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

# COCO
dataset_coco = {"name": "coco",
                "im_dir": "/home/ps/Guo/Project/sam-hq-main/eval_coco_ours/data/COCO/train2017",
                "gt_dir": "/home/ps/Guo/Project/sam-hq-main/eval_coco_ours/data/COCO/mask_train",
                "im_ext": ".jpg",
                "gt_ext": ".jpg"}

# Polyp
dataset_polyp = {"name": "polyp-train",
                "im_dir": "./data/PolySeg/TrainDataset/image",
                "gt_dir": "./data/PolySeg/TrainDataset/masks",
                "edge_dir": "./data/PolySeg/TrainDataset/b-map",
                "im_ext": ".png",
                "gt_ext": ".png"}


# low light  
dataset_dis_low = {"name": "DIS5K-TR",
                "im_dir": "./data/DIS5K/DIS-TR-1536/lowlight_im",
                "gt_dir": "./data/DIS5K/DIS-TR-1536/gt",
                "edge_dir": "./data/DIS5K/DIS-TR-1536/b-map",
                "im_ext": ".png",
                "gt_ext": ".png"}


dataset_fss_low = {"name": "FSS-low",
                "im_dir": "./data/cascade_psp/fss_all-low/fss_all-low",
                "gt_dir": "./data/cascade_psp/fss_all-low/fss_all-low",
                "edge_dir": "./data/cascade_psp/fss_all-low/b_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_duts_low = {"name": "DUTS-TR-low",
                "im_dir": "./data/cascade_psp/DUTS-TR-low/DUTS-TR-low",
                "gt_dir": "./data/cascade_psp/DUTS-TR-low/DUTS-TR-low",
                "edge_dir": "./data/cascade_psp/DUTS-TR-low/b_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_duts_te_low = {"name": "DUTS-TE",
                "im_dir": "./data/cascade_psp/DUTS-TE-low/DUTS-TE-low",
                "gt_dir": "./data/cascade_psp/DUTS-TE-low/DUTS-TE-low",
                "edge_dir": "./data/cascade_psp/DUTS-TE-low/b_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_ecssd_low = {"name": "ECSSD-low",
                "im_dir": "./data/cascade_psp/ecssd-low/ecssd-low",
                "gt_dir": "./data/cascade_psp/ecssd-low/ecssd-low",
                "edge_dir": "./data/cascade_psp/ecssd-low/b_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_msra_low = {"name": "MSRA10K-low",
                "im_dir": "./data/cascade_psp/MSRA_10K-low/MSRA_10K-low",
                "gt_dir": "./data/cascade_psp/MSRA_10K-low/MSRA_10K-low",
                "edge_dir": "./data/cascade_psp/MSRA_10K-low/b_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dsataset_thin_low = {"name": "ThinObject5k-TR-low",
                "im_dir": "./data/thin_object_detection/ThinObject5K_low/images_train",
                "gt_dir": "./data/thin_object_detection/ThinObject5K_low/images_train",
                "edge_dir": "./data/thin_object_detection/ThinObject5K_low/b_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


# hq
dataset_dis = {"name": "DIS5K-TR",
                "im_dir": "./data/DIS5K/DIS-TR/im",
                "gt_dir": "./data/DIS5K/DIS-TR/gt",
                "edge_dir": "./data/DIS5K/DIS-TR/b-map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_thin = {"name": "ThinObject5k-TR",
                "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                "edge_dir": "./data/thin_object_detection/ThinObject5K/edges_train",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_fss = {"name": "FSS",
                 "im_dir": "./data/cascade_psp/fss_all",
                 "gt_dir": "./data/cascade_psp/fss_all",
                 "edge_dir": "./data/cascade_psp/fss_all-b",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}


dataset_duts = {"name": "DUTS-TR",
                "im_dir": "./data/cascade_psp/DUTS-TR",
                "gt_dir": "./data/cascade_psp/DUTS-TR",
                "edge_dir": "./data/cascade_psp/DUTS-TR-b",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_duts_te = {"name": "DUTS-TE",
                "im_dir": "./data/cascade_psp/DUTS-TE",
                "gt_dir": "./data/cascade_psp/DUTS-TE",
                "edge_dir": "./data/cascade_psp/DUTS-TE-b",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_ecssd = {"name": "ECSSD",
                "im_dir": "./data/cascade_psp/ecssd",
                "gt_dir": "./data/cascade_psp/ecssd",
                "edge_dir": "./data/cascade_psp/ecssd-b",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_msra = {"name": "MSRA10K",
                "im_dir": "./data/cascade_psp/MSRA_10K",
                "gt_dir": "./data/cascade_psp/MSRA_10K",
                "edge_dir": "./data/cascade_psp/MSRA_10K-b",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

# degraded low resolution datasets
dataset_msra_lr = {"name": "MSRA10K-LR",
                "im_dir": "./data/cascade_psp/MSRA_10K-RealLR/lr",
                "gt_dir": "./data/cascade_psp/MSRA_10K",
                "edge_dir": "./data/cascade_psp/MSRA_10K-b",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_thin_lr = {"name": "ThinObject5k-LR-TR",
                "im_dir": "./data/thin_object_detection/ThinObject5K-RealLR/images_train/lr",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                "edge_dir": "./data/thin_object_detection/ThinObject5K/edges_train",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


# valid set
# low light 
dataset_dislow_val = {"name": "DIS5K-VD-low",
                "im_dir": "./data/DIS5K/DIS-VD-1536/lowlight_im",
                "gt_dir": "./data/DIS5K/DIS-VD-1536/gt",
                "im_ext": ".png",
                "gt_ext": ".png"}


dataset_thinlow_val = {"name": "ThinObject5k-TE-low",
                "im_dir": "./data/thin_object_detection/ThinObject5K_low/images_test",
                "gt_dir": "./data/thin_object_detection/ThinObject5K_low/images_test",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_lisu_val = {"name": "LISU-low",
                "im_dir": "./data/LISU/LISU_LLRGBD_real/LISU-2-8-9-11/img",
                "gt_dir": "./data/LISU/LISU_LLRGBD_real/LISU-2-8-9-11/gt",
                "im_ext": ".png",
                "gt_ext": ".png"}


# camo
dataset_camo_val = {"name": "CAMO",
                "im_dir": "./data/COD/TestDataset/CAMO/Imgs",
                "gt_dir": "./data/COD/TestDataset/CAMO/GT",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_cha_val = {"name": "CHAMELEON",
                "im_dir": "./data/COD/TestDataset/CHAMELEON/Imgs",
                "gt_dir": "./data/COD/TestDataset/CHAMELEON/GT",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_cod10k_val= {"name": "COD10K",
                "im_dir": "./data/COD/TestDataset/COD10K/Imgs",
                "gt_dir": "./data/COD/TestDataset/COD10K/GT",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_NC4K_val = {"name": "NC4K",
                "im_dir": "./data/COD/TestDataset/NC4K/Imgs",
                "gt_dir": "./data/COD/TestDataset/NC4K/GT",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

# hq
dataset_coift_val = {"name": "COIFT",
                "im_dir": "./data/thin_object_detection/COIFT/images",
                "gt_dir": "./data/thin_object_detection/COIFT/masks",
                "edge_dir": "./data/thin_object_detection/COIFT/edge_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_hrsod_val = {"name": "HRSOD",
                "im_dir": "./data/thin_object_detection/HRSOD/images",
                "gt_dir": "./data/thin_object_detection/HRSOD/masks",
                "edge_dir": "./data/thin_object_detection/HRSOD/edge_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_thin_val = {"name": "ThinObject5k-TE",
                "im_dir": "./data/thin_object_detection/ThinObject5K/test/images",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/test/masks",
                "edge_dir": "./data/thin_object_detection/ThinObject5K/edges_test",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_dis_val = {"name": "DIS5K-VD",
                "im_dir": "./data/DIS5K/DIS-VD/images",
                "gt_dir": "./data/DIS5K/DIS-VD/masks",
                "edge_dir": "./data/DIS5K/DIS-VD/edge_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


# poly seg
dataset_colondb_val = {"name": "CVC-ColonDB",
                "im_dir": "./data/PolySeg/TestDataset/CVC-ColonDB/images",
                "gt_dir": "./data/PolySeg/TestDataset/CVC-ColonDB/masks",
                "im_ext": ".png",
                "gt_ext": ".png"}
dataset_cvc300_val = {"name": "CVC-300",
                "im_dir": "./data/PolySeg/TestDataset/CVC-300/images",
                "gt_dir": "./data/PolySeg/TestDataset/CVC-300/masks",
                "im_ext": ".png",
                "gt_ext": ".png"}
dataset_clinicdb_val = {"name": "CVC-ClinicDB",
                "im_dir": "./data/PolySeg/TestDataset/CVC-ClinicDB/images",
                "gt_dir": "./data/PolySeg/TestDataset/CVC-ClinicDB/masks",
                "im_ext": ".png",
                "gt_ext": ".png"}
dataset_kvasir_val = {"name": "Kvasir",
                "im_dir": "./data/PolySeg/TestDataset/Kvasir/images",
                "gt_dir": "./data/PolySeg/TestDataset/Kvasir/masks",
                "im_ext": ".png",
                "gt_ext": ".png"}


dataset_etis_val = {"name": "ETIS",
                "im_dir": "./data/PolySeg/TestDataset/ETIS/images",
                "gt_dir": "./data/PolySeg/TestDataset/ETIS/masks",
                "im_ext": ".png",
                "gt_ext": ".png"}


dataset_coco_val = {"name": "coco-val",
                    "im_dir": "/mnt/nvme1n1/Guo/dataset/RobustSeg/test/COCO/images",
                    "gt_dir": "/mnt/nvme1n1/Guo/dataset/RobustSeg/test/COCO/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

dataset_coco_NS_val = {"name": "coco-ns-val",
                    "im_dir": "data/VNS-COCO/images",
                    "gt_dir": "data/VNS-COCO/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

# defect detection
datasset_cds2k = {"name": "cds2k",
                "im_dir": "/home/ps/Guo/dataset/CDS2K/Image",
                "gt_dir": "/home/ps/Guo/dataset/CDS2K/GroundTruth",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

# transparent detection
datasset_trans10K = {"name": "trans10k",
                "im_dir": "./data/Trans10K/test/images",
                "gt_dir": "./data/Trans10K/test/masks",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_GDD_val = {
    "name": "GDD_val",
    "im_dir": "/home/ps/Guo/dataset/GDD/test/image",
    "gt_dir": "/home/ps/Guo/dataset/GDD/test/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}


# low resolution images
dataset_thin_lr_val = {"name": "thin-lr-val",
                "im_dir": "./data/thin_object_detection/ThinObject5K-RealLR/images_test/lr/lr",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_thin_val_lr_osediff = {"name": "thin-lr-osediff",
                "im_dir": "./data/thin_object_detection/ThinObject5K-RealLR-OSEdiff/images_test",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_duts_te_lr = {"name": "DUTS-TE-LR",
                "im_dir": "./data/cascade_psp/DUTS-TE-LR",
                "gt_dir": "./data/cascade_psp/DUTS-TE-LR",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

# dataset_msra_lr = {"name": "MSRA-LR",
#                 "im_dir": "./data/cascade_psp/MSRA_10K-LR",
#                 "gt_dir": "./data/cascade_psp/MSRA_10K-LR",
#                 "im_ext": ".jpg",
#                 "gt_ext": ".png"}


dataset_ecssd_lr = {"name": "ecssd-LR",
                # "im_dir": "./data/cascade_psp/ecssd-RealLR-sf2/lr",
                "im_dir": "./data/cascade_psp/ecssd-RealLR/lr",
                "gt_dir": "./data/cascade_psp/ecssd/",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_ecssd_lr_ose_diff = {"name": "ecssd-LR-OSEDiff",
                # "im_dir": "./data/cascade_psp/ecssd-RealLR-OSEDiff-orisize-sf2",
                "im_dir": "./data/cascade_psp/ecssd-LR_ratio4-OSEDiff-orisize/",
                "gt_dir": "./data/cascade_psp/ecssd/",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_ecssd_ose_diff_sr = {"name": "ecssd-SR-OSEDiff",
                "im_dir": "./data/cascade_psp/ecssd-OSEDiff-SR",
                "gt_dir": "./data/cascade_psp/ecssd",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


dataset_dis_val_lr = {"name": "DIS-VD-LR",
                "im_dir": "./data/DIS5K/DIS-VD-RealLR/lr",
                "gt_dir": "./data/DIS5K/DIS-VD/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png"}


# dataset_duts_te = {"name": "DUTS-TE",
#                 "im_dir": "./data/cascade_psp/DUTS-TE",
#                 "gt_dir": "./data/cascade_psp/DUTS-TE",
#                 "im_ext": ".jpg",
#                 "gt_ext": ".png"}

# def build_nosl():
#     train_datasets=[dataset_cod, dataset_dis_low, dsataset_thin_low, dataset_fss_low,  dataset_duts_low, dataset_duts_te_low, dataset_ecssd_low, dataset_msra_low]
#     valid_datasets = [dataset_camo_val,  dataset_cod10k_val, dataset_dislow_val,  dataset_thinlow_val]
#     # valid_datasets = [dataset_dislow_val,  dataset_thinlow_val]
#     # dataset_camo_val,  dataset_cod10k_val, dataset_dislow_val, 
#     return train_datasets, valid_datasets

def build_nosl():
    train_datasets=[dataset_cod, dataset_polyp,  dataset_dis_low, dsataset_thin_low, dataset_fss_low]
    # all
    valid_datasets = [dataset_camo_val, dataset_cod10k_val, dataset_kvasir_val, dataset_clinicdb_val, dataset_dislow_val,  dataset_thinlow_val, dataset_NC4K_val, dataset_colondb_val, dataset_etis_val, datasset_cds2k, dataset_coco_val]
    # valid_datasets = [dataset_NC4K_val, dataset_etis_val, dataset_dislow_val]
    # seen
    # valid_datasets = [dataset_camo_val, dataset_cod10k_val, dataset_kvasir_val, dataset_clinicdb_val, dataset_dislow_val,  dataset_thinlow_val]
    # unseen
    # valid_datasets = [dataset_NC4K_val, dataset_colondb_val, dataset_etis_val, datasset_cds2k]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val, dataset_cod10k_val]
    
    return train_datasets, valid_datasets

def build_cod_hq():
    train_datasets=[dataset_cod, dataset_dis, dataset_thin, dataset_fss, dataset_ecssd]
    valid_datasets = [dataset_camo_val, dataset_cod10k_val, dataset_kvasir_val, dataset_clinicdb_val, dataset_dislow_val,  dataset_thinlow_val, dataset_NC4K_val, dataset_colondb_val, dataset_etis_val,  dataset_lisu_val, datasset_cds2k]

    return train_datasets, valid_datasets

def build_cod():
    train_datasets=[dataset_cod]
    valid_datasets = [dataset_camo_val,  dataset_cod10k_val, dataset_NC4K_val]
    valid_datasets = [dataset_NC4K_val, dataset_etis_val, dataset_dislow_val]
    return train_datasets, valid_datasets

def build_hq():
    train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 
    return train_datasets, valid_datasets

def build_hqtrainval():
    train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra, dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val]
    valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 
    return train_datasets, valid_datasets

def build_hqns():
    train_datasets=[dataset_cod, dataset_polyp, dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra, dataset_dis_low, dataset_fss_low, dsataset_thin_low]
    # valid_datasets = [dataset_cha_val, dataset_camo_val,  dataset_cod10k_val, dataset_NC4K_val, dataset_dislow_val, dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val]
    valid_datasets = [dataset_kvasir_val, dataset_camo_val,  dataset_dislow_val,  dataset_thinlow_val]
    return train_datasets, valid_datasets

def build_lisu():
    train_dataset = [dataset_lisu_val]
    valid_dataset = [dataset_lisu_val]
    return train_dataset, valid_dataset

def build_dark():
    train_dataset = [dataset_dis_low, dataset_fss_low, dsataset_thin_low]
    valid_dataset = [dataset_dislow_val, dataset_thinlow_val, dataset_coco_val]
    valid_dataset = [dataset_dislow_val]
    return train_dataset, valid_dataset


def build_normal():
    train_dataset = [dataset_dis, dataset_fss, dsataset_thin_low]
    valid_dataset = [dataset_NC4K_val]
    return train_dataset, valid_dataset


def build_poly():
    train_datasets = [dataset_polyp]
    valid_datasets = [dataset_kvasir_val, dataset_clinicdb_val, dataset_colondb_val, dataset_etis_val]
    valid_datasets = [dataset_etis_val, dataset_dislow_val]
    return train_datasets, valid_datasets

def build_vis():
    valid_datasets = [dataset_cod10k_val, dataset_lisu_val]
    return valid_datasets, valid_datasets

def build_coco():
    train_dataset = [dataset_coco]
    valid_dataset = [dataset_coco_val]
    # valid_dataset = [dataset_coco_NS_val]
    return train_dataset, valid_dataset

def build_defect():
    valid_dataset = [datasset_cds2k]
    return valid_dataset, valid_dataset

def build_transp():
    valid_dataset = [dataset_GDD_val]
    return valid_dataset, valid_dataset


def build_thin_lr():
    valid_dataset = [dataset_thin_val_lr_osediff]
    return valid_dataset, valid_dataset


def build_thin():
    valid_dataset = [dataset_thin_val, dataset_thin_lr_val]
    return valid_dataset, valid_dataset


# def build_duts_lr():
#     valid_dataset = [dataset_duts_te_lr]
#     return valid_dataset, valid_dataset

def build_duts():
    valid_dataset = [dataset_duts_te, dataset_duts_te_lr]
    return valid_dataset, valid_dataset

# def build_msra_lr():
#     valid_dataset = [dataset_msra_lr]
#     return valid_dataset, valid_dataset

# def build_ecssd_lr():
#     valid_dataset = [dataset_ecssd_lr]
#     return valid_dataset, valid_dataset

def build_ecssd():
    valid_dataset = [dataset_ecssd_lr, dataset_ecssd_lr_ose_diff,  dataset_ecssd_ose_diff_sr, dataset_ecssd]
    valid_dataset = [dataset_ecssd_lr, dataset_ecssd_lr_ose_diff]
    valid_dataset = [dataset_ecssd, dataset_ecssd_lr_ose_diff, dataset_ecssd_ose_diff_sr]

    return valid_dataset, valid_dataset

def build_msra():
    valid_dataset = [dataset_msra, dataset_msra_lr]
    return valid_dataset, valid_dataset

def build_lowres_dataset():
    train_dataset = [dataset_msra_lr, dataset_thin_lr]
    valid_dataset = [dataset_thin_lr_val, dataset_ecssd_lr]
    return train_dataset, valid_dataset


def build_dis():
    valid_dastaset = [dataset_dis_val]
    return valid_dastaset, valid_dastaset


dataset_registry = {
    "ns": build_nosl,
    "cod": build_cod,
    "hq": build_hq,
    "hqtrainval": build_hqtrainval,
    "hqns": build_hqns,
    "lisu": build_lisu,
    "poly": build_poly,
    "vis": build_vis,
    "coco": build_coco,
    "dark": build_dark,
    "defect": build_defect,
    "trans": build_transp,
    "cod-hq": build_cod_hq,
    "thin-lr": build_thin_lr,
    "thin": build_thin,
    "duts": build_duts,
    "msra": build_msra,
    "ecssd": build_ecssd,
    "dis": build_dis,
    "lr": build_lowres_dataset,
    "normal": build_normal
}
