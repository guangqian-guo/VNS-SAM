# train set
# COD
dataset_cod = {"name": "COD",
                "im_dir": "./data/COD/TrainDataset/Imgs",
                "gt_dir": "./data/COD/TrainDataset/GT",
                "edge_dir": "./data/COD/TrainDataset/Edge",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

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

dsataset_thin_low = {"name": "ThinObject5k-TR-low",
                "im_dir": "./data/thin_object_detection/ThinObject5K_low/images_train",
                "gt_dir": "./data/thin_object_detection/ThinObject5K_low/images_train",
                "edge_dir": "./data/thin_object_detection/ThinObject5K_low/b_map",
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

# poly seg
dataset_colondb_val = {"name": "CVC-ColonDB",
                "im_dir": "./data/PolySeg/TestDataset/CVC-ColonDB/images",
                "gt_dir": "./data/PolySeg/TestDataset/CVC-ColonDB/masks",
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


# defect detection
datasset_cds2k = {"name": "cds2k",
                "im_dir": "/home/ps/Guo/dataset/CDS2K/Image",
                "gt_dir": "/home/ps/Guo/dataset/CDS2K/GroundTruth",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

def build_nosl():
    train_datasets=[dataset_cod, dataset_polyp,  dataset_dis_low, dsataset_thin_low, dataset_fss_low]
    valid_datasets = [dataset_camo_val, dataset_cod10k_val, dataset_kvasir_val, dataset_clinicdb_val, dataset_dislow_val,  dataset_thinlow_val, dataset_NC4K_val, dataset_colondb_val, dataset_etis_val, datasset_cds2k]
    
    return train_datasets, valid_datasets

dataset_registry = {
    "ns": build_nosl
}