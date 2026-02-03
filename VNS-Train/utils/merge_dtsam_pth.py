import torch


decoder_ckpt = torch.load('../VNS-SAM-Train/work-dir/major/DTSAM_trained_on_dark/epoch_11.pth')
sam_ckpt = torch.load('/home/ps/Guo/Project/sam-hq-main/train/pretrained_checkpoint/sam_vit_l_0b3195.pth')

for key in decoder_ckpt.keys():
    new_key = "mask_decoder." + key
    if new_key in sam_ckpt.keys():
        print(key)
        sam_ckpt[new_key] = decoder_ckpt[key]
        
torch.save(sam_ckpt, '../VNS-SAM-Train/work-dir/major/DTSAM_trained_on_dark/DT-SAM.pth')