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
                "edge_dir": "./data/DIS5K/DIS-TR/b-map",
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
                "im_dir": "./data/LISU/LISU_LLRGBD_real/LISU-8-9-10-11/img",
                "gt_dir": "./data/LISU/LISU_LLRGBD_real/LISU-8-9-10-11/gt",
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
                "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                "edge_dir": "./data/thin_object_detection/HRSOD/edge_map",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_thin_val = {"name": "ThinObject5k-TE",
                "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                "edge_dir": "./data/thin_object_detection/ThinObject5K/edges_test",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_dis_val = {"name": "DIS5K-VD",
                "im_dir": "./data/DIS5K/DIS-VD/im",
                "gt_dir": "./data/DIS5K/DIS-VD/gt",
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


dataset_coco_val = {"name": "coco",
                "im_dir": "/home/ps/Guo/Project/sam-hq-main/eval_coco_ours/data/COCO/val2017",
                "gt_dir": "/home/ps/Guo/Project/sam-hq-main/eval_coco_ours/data/COCO/mask_val",
                "im_ext": ".jpg",
                "gt_ext": ".jpg"}


# def build_nosl():
#     train_datasets=[dataset_cod, dataset_dis_low, dsataset_thin_low, dataset_fss_low,  dataset_duts_low, dataset_duts_te_low, dataset_ecssd_low, dataset_msra_low]
#     valid_datasets = [dataset_camo_val,  dataset_cod10k_val, dataset_dislow_val,  dataset_thinlow_val]
#     # valid_datasets = [dataset_dislow_val,  dataset_thinlow_val]
#     # dataset_camo_val,  dataset_cod10k_val, dataset_dislow_val, 
#     return train_datasets, valid_datasets

def build_nosl():
    train_datasets=[dataset_cod, dataset_polyp,  dataset_dis_low, dsataset_thin_low, dataset_fss_low]
    valid_datasets = [dataset_kvasir_val, dataset_camo_val,  dataset_dislow_val,  dataset_thinlow_val]
    valid_datasets = [dataset_colondb_val, dataset_etis_val, dataset_NC4K_val, dataset_lisu_val]
    
    # valid_datasets = [dataset_NC4K_val, dataset_lisu_val]
    
    # valid_datasets = [ dataset_cod10k_val]
    # valid_datasets = [dataset_dislow_val,  dataset_thinlow_val]
    # dataset_camo_val,  dataset_cod10k_val, dataset_dislow_val, 
    return train_datasets, valid_datasets


def build_cod():
    train_datasets=[dataset_cod]
    valid_datasets = [dataset_cha_val, dataset_camo_val,  dataset_cod10k_val, dataset_NC4K_val]
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

def build_poly():
    valid_datasets = [dataset_kvasir_val, dataset_colondb_val, dataset_clinicdb_val,  dataset_etis_val ,dataset_cvc300_val   ]
    # valid_datasets = [dataset_colondb_val ]
    return valid_datasets, valid_datasets

def build_vis():
    valid_datasets = [dataset_cod10k_val, dataset_lisu_val]
    return valid_datasets, valid_datasets


def build_coco():
    train_dataset = [dataset_coco]
    valid_dataset = [dataset_coco_val]
    return train_dataset, valid_dataset


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
}



