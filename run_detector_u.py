# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License
# Modified by Cristhian Moya Mota — default detector set to MaskedPatchMorphLOFContrastAgg

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--file_list', type=str, default='Datasets/File_List/TSB-AD-U-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
    parser.add_argument('--save', action='store_true', help='Save anomaly scores (.npy) and metrics (.csv) to disk')
    parser.add_argument('--AD_Name', type=str, default='MaskedPatchMorphLOFContrastAgg')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for reproducibility')
    args = parser.parse_args()

    # seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    if args.save:
        os.makedirs(args.save_dir, exist_ok = True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)

    write_csv = []
    for filename in file_list:
        if os.path.exists(target_dir+'/'+filename.split('.')[0]+'.npy'): continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        if args.AD_Name in Semisupervise_AD_Pool:
            output, timing = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
        elif args.AD_Name in Unsupervise_AD_Pool:
            output, timing = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
        else:
            raise Exception(f"{args.AD_Name} is not defined")

        end_time = time.time()
        # Total: whole model call; Train/Infer: fit() and decision_function() alone.
        total_time = end_time - start_time
        train_time = timing['train'] if timing else 0.0
        infer_time = timing['infer'] if timing else 0.0

        if isinstance(output, np.ndarray):
            logging.info(f'Success at {filename} using {args.AD_Name} | Total: {total_time:.3f}s Train: {train_time:.3f}s Infer: {infer_time:.3f}s at length {len(label)}')
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)
        else:
            logging.error(f'At {filename}: '+output)

        ### whether to save the evaluation result
        if args.save:
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except:
                list_w = [0]*9
            list_w.insert(0, infer_time)
            list_w.insert(0, train_time)
            list_w.insert(0, total_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'Infer (s)')
            col_w.insert(0, 'Train (s)')
            col_w.insert(0, 'Total (s)')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)
