import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b":"pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        
        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


def inference(opt):
    model = MaskDecoderHQ(opt.model_type) 
    # Network = select_model(opt.model)
    # model = Network(imagenet_pretrained=False)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    
    # load_weights
    weights = torch.load(opt.pth_path)
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    model.load_state_dict(weights)
    
    model.cuda()
    model.eval()
    for _data_name in ['CHAMELEON','CAMO', 'COD10K', 'NC4K']: #['CAMO', 'COD10K', 'CHAMELEON']:
        print('==================Processing {}===================='.format(_data_name))
        data_path = './Dataset/COD/TestDataset/{}'.format(_data_name)
        save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        
        for i in tqdm(range(test_loader.size)):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            
            # res = res2
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # print('> {} - {}'.format(_data_name, name))
            # plt.imshow(res)
            # plt.savefig(save_path+name)
            # misc.imageio.imwrite(save_path+name, res)
            # If `mics` not works in your environment, please comment it and then use CV2
            cv2.imwrite(save_path+name,res*255)

def evaluate(args):
    
    # metrics
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    mask_dataset_list = ['CHAMELEON','CAMO', 'COD10K', 'NC4K']
    # mask_name_list = sorted(os.listdir(args.mask_dir))
    results_list = []
    
    pred_dir = './res/{}/'.format(opt.pth_path.split('/')[-2])
    for i, dataset in enumerate(mask_dataset_list):
        # metrics
        WFM = py_sod_metrics.WeightedFmeasure()
        SM = py_sod_metrics.Smeasure()
        EM = py_sod_metrics.Emeasure()
        MAE = py_sod_metrics.MAE()
        print(f"================{datetime.now()} [{i}] Processing {dataset}================")
        logging.info(f"[{i}] Processing {dataset}")
        mask_dataset_dir = os.path.join(args.mask_dir, dataset)
        pred_dataset_dir = os.path.join(pred_dir, dataset)
        pred_name_list = sorted(os.listdir(pred_dataset_dir))
        for j in tqdm(range(len(pred_name_list))):
            # print(f"[{i}] Processing {mask_name}...")
            mask_name = pred_name_list[j]
            mask_path = os.path.join(mask_dataset_dir+'/GT', mask_name)
            pred_path = os.path.join(pred_dataset_dir, mask_name)        
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            MAE.step(pred=pred, gt=mask)

        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]

        curr_results = {
            "MAE": mae,
            "Smeasure": sm,
            "wFmeasure": wfm,
            # E-measure for sod
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max()
        }
        
        results_list.append(curr_results)
        print(curr_results)
        logging.info(str(curr_results))
   
    return results_list

def generate_benchmark_table(results, args):
    results_log = './res/{}/results.txt'.format(args.pth_path.split('/')[-2])
    
    for k in range(len(results)):
        result = results[k]
        # print(line.split('Model:')[1].split(') Smeasure')[0], model_lst[i])
        S_measure = '%.3f'%result['Smeasure']
        w_F = '%.3f'%result['wFmeasure']
        mean_E_m = '%.3f'%result['meanEm']
        MAE = round(result['MAE'],3)
        res_latex = '& {}   & {}   & {}   & {}'.format(S_measure, mean_E_m, w_F, MAE)
        print(res_latex, end='\n')
        with open(results_log, 'a') as f:
            f.write(res_latex+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='select a model')
    parser.add_argument('--testsize', type=int, default=1024, help='testing size')
    parser.add_argument('--pth-path', type=str, default='./snapshot/SINet_V2/Net_epoch_best.pth')
    parser.add_argument('--mask-dir', type=str, default='data/COD/TestDataset/', help='ground truth map')
    parser.add_argument('--pred-dir', type=str, default='', help='predicted map')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    opt = parser.parse_args()
    
    log_path = './res/{}/'.format(opt.pth_path.split('/')[-2])
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_path + 'test.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(f"Network: {opt.pth_path.split('/')[-2]}")
    logging.info("Network-Test")
    
    # inference
    logging.info("Beginning inference!")
    print('========beginning inference!========')
    inference(opt)
    logging.info("Inference ending!")
    
    if opt.evaluate:
        logging.info("Beginning evaluate!")
        print('=======beginning evaluate!=========')
        results = evaluate(opt)
        print('======={} generating latex label!========'.format(datetime.now()))
        generate_benchmark_table(results, opt)
        logging.info("Evaluating ending!")
        