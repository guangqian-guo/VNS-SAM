from statistics import mode
from unittest import result
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2

from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
import sys
import unittest
from pprint import pprint
import py_sod_metrics
import json
import logging


'''
CUDA_VISIBLE_DEVICES=4 python MyTesting_COD.py  --model Camprompterv6 --pth-path ./snapshot/COD/Camprompterv6Net_epoch_best.pth --mask-dir ./Dataset/COD/TestDataset/
python MyTesting_COD.py --pred-dir work_dirs/sam-l-edge-enhancev4-23/hq_sam/ --mask-dir data/COD/TestDataset
'''

def evaluate(args):
    # metrics
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    mask_dataset_list = ['CHAMELEON','CAMO', 'COD10K', 'NC4K']
    # mask_name_list = sorted(os.listdir(args.mask_dir))
    results_list = []
    
    pred_dir = args.pred_dir
    for i, dataset in enumerate(mask_dataset_list):
        # metrics
        WFM = py_sod_metrics.WeightedFmeasure()
        SM = py_sod_metrics.Smeasure()
        EM = py_sod_metrics.Emeasure()
        MAE = py_sod_metrics.MAE()
        
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        
        print(f"================{datetime.now()} [{i}] Processing {dataset}================")
        logging.info(f"[{i}] Processing {dataset}")
        mask_dataset_dir = os.path.join(args.mask_dir, dataset)
        # mask_dataset_dir = args.mask_dir
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
            
            intersection, union, target = intersectionAndUnion(pred, mask)
            
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        iou_dataset = intersection_meter.sum / (union_meter.sum + 1e-10)
        
        
        
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
            "maxEm": em["curve"].max(),
            "miou": iou_dataset
        }
        
        results_list.append(curr_results)
        print(curr_results)
        logging.info(str(curr_results))
    
    return results_list

def intersectionAndUnion(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    
    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()
    
    return area_intersection, area_union, area_target

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def eval_miou(args):
#     mask_dataset_list = ['CHAMELEON','CAMO', 'COD10K', 'NC4K']
#     # mask_name_list = sorted(os.listdir(args.mask_dir))
#     results_list = []
    
#     pred_dir = args.pred_dir
#     for i, dataset in enumerate(mask_dataset_list):
#         print(f"================{datetime.now()} [{i}] Processing {dataset}================")
#         logging.info(f"[{i}] Processing {dataset}")
#         mask_dataset_dir = os.path.join(args.mask_dir, dataset)
#         pred_dataset_dir = os.path.join(pred_dir, dataset)
#         pred_name_list = sorted(os.listdir(pred_dataset_dir))
#         for j in tqdm(range(len(pred_name_list))):
#             mask_name = pred_name_list[j]

#             mask_path = os.path.join(mask_dataset_dir+'/GT', mask_name)
#             pred_path = os.path.join(pred_dataset_dir, mask_name)        
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            
            
def generate_benchmark_table(results, args):
    results_log = './work_dirs/{}/results.txt'.format(args.pred_dir.split('/')[-1])
    
    for k in range(len(results)):
        result = results[k]
        # print(line.split('Model:')[1].split(') Smeasure')[0], model_lst[i])
        S_measure = '%.3f'%result['Smeasure']
        w_F = '%.3f'%result['wFmeasure']
        mean_E_m = '%.3f'%result['meanEm']
        MAE = round(result['MAE'],3)
        miou = '%.3f'%result['miou']
        res_latex = '& {}   & {}   & {}   & {} & {}'.format(S_measure, mean_E_m, w_F, MAE, miou)
        print(res_latex, end='\n')
        with open(results_log, 'a') as f:
            f.write(res_latex+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask-dir', type=str, default='data/COD/TestDataset/', help='ground truth map')
    parser.add_argument('--pred-dir', type=str, default='', help='predicted map')
    opt = parser.parse_args()
    
    log_path = opt.pred_dir
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_path + 'test.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # logging.info(f"Network: {opt.pth_path.split('/')[-2]}")
    # logging.info("Network-Test")
    
   
    logging.info("Beginning evaluate!")
    print('=======beginning evaluate!=========')
    results = evaluate(opt)
    print('======={} generating latex label!========'.format(datetime.now()))
    generate_benchmark_table(results, opt)
    logging.info("Evaluating ending!")
    