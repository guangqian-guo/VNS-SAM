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

'''

def evaluate(args):
    # metrics
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    mask_dataset_list = ['Kvasir', 'CVC-ColonDB', 'CVC-ClinicDB',  'ETIS', 'CVC-300']
    # mask_name_list = sorted(os.listdir(args.mask_dir))
    results_list = []
    
    pred_dir = args.pred_dir
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

            mask_path = os.path.join(mask_dataset_dir+'/masks', mask_name)
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
    results_log = './work_dirs/{}/results.txt'.format(args.pred_dir.split('/')[-1])
    
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
    parser.add_argument('--mask-dir', type=str, default='./data/PolySeg/TestDataset/', help='ground truth map')
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
    