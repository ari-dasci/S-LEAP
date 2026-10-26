# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License
# Modified by Cristhian Moya Mota — default detector set to MaskedPatchMorphLOFContrastAgg

import pandas as pd
import numpy as np
import torch
import gc
import random
import argparse
import time
import os
import logging
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

# Enable/Disable TF32 for matrix multiplications (e.g., torch.matmul)
torch.backends.cuda.matmul.allow_tf32 = True

# Enable/Disable TF32 for convolution operations (e.g., cuDNN)
torch.backends.cudnn.allow_tf32 = True


print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()

    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str,
                        default='Datasets/TSB-AD-M/')
    parser.add_argument('--file_list',   type=str,  default='Datasets/File_List/TSB-AD-M-Eva.csv',
                        help='TSB-AD-M-Eva.csv (benchmark subset, 180) or TSB-AD-M.csv (full, 200)')
    parser.add_argument('--score_dir',   type=str,
                        default='eval/score/multi/')
    parser.add_argument('--save_dir',    type=str,
                        default='eval/metrics/multi/')
    parser.add_argument('--save',        action='store_true',
                        help='Save anomaly scores (.npy) and metrics (.csv) to disk')
    parser.add_argument('--AD_Name',     type=str,
                        default='MaskedPatchMorphLOFContrastAgg')
    parser.add_argument('--seed',        type=int,  default=2024,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Seeding for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    if args.save:
        os.makedirs(target_dir, exist_ok=True)

    logging.basicConfig(
        filename=f'{target_dir}/000_run_{args.AD_Name}.log' if args.save else os.devnull,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)
    print(f'Files to process: {len(file_list)}\n')

    all_metrics = []

    for filename in file_list:
        score_path = target_dir + '/' + filename.split('.')[0] + '.npy'
        if args.save and os.path.exists(score_path):
            print(f'[SKIP] {filename} (score already exists)')
            continue
        print(f'Processing: {filename}')

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        if args.AD_Name in Semisupervise_AD_Pool:
            output, timing = run_Semisupervise_AD(
                args.AD_Name, data_train, data, **Optimal_Det_HP)
        elif args.AD_Name in Unsupervise_AD_Pool:
            output, timing = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
        else:
            raise Exception(f"{args.AD_Name} is not defined")

        # Total: whole model call; Train/Infer: fit() and decision_function() alone.
        total_time = time.time() - start_time
        train_time = timing['train'] if timing else 0.0
        infer_time = timing['infer'] if timing else 0.0

        if not isinstance(output, np.ndarray):
            print(f'  [ERROR] {output}')
            logging.error(f'At {filename}: {output}')
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if args.save:
            np.save(score_path, output)

        try:
            evaluation_result = get_metrics(
                output, label, slidingWindow=slidingWindow)
            metrics_str = '  '.join(
                f'{k}: {v:.4f}' for k, v in evaluation_result.items())
            print(f'  [Total {total_time:.1f}s | Train {train_time:.1f}s | Infer {infer_time:.1f}s] {metrics_str}')

            row = {'file': filename, 'Total (s)': total_time,
                   'Train (s)': train_time, 'Infer (s)': infer_time,
                   **evaluation_result}
            all_metrics.append(row)
        except Exception as e:
            print(f'  [EVAL ERROR] {e}')

        logging.info(
            f'Success at {filename} | Total: {total_time:.3f}s | Train: {train_time:.3f}s | '
            f'Infer: {infer_time:.3f}s | length: {len(label)}')

        gc.collect()
        torch.cuda.empty_cache()

        if args.save and all_metrics:
            os.makedirs(args.save_dir, exist_ok=True)
            pd.DataFrame(all_metrics).to_csv(
                f'{args.save_dir}/{args.AD_Name}.csv', index=False)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        print('\n' + '=' * 60)
        print(f'SUMMARY — {args.AD_Name}  ({len(df_metrics)} files)')
        print('=' * 60)
        time_cols = ['Total (s)', 'Train (s)', 'Infer (s)']
        numeric_cols = df_metrics.select_dtypes(
            include=np.number).columns.drop(time_cols, errors='ignore')
        print(df_metrics[numeric_cols].mean().to_string())
        print(f'\nMean per series — Total: {df_metrics["Total (s)"].mean():.3f}s | '
              f'Train: {df_metrics["Train (s)"].mean():.3f}s | '
              f'Infer: {df_metrics["Infer (s)"].mean():.3f}s')
        print(f'Wall clock: {time.time() - Start_T:.1f}s')
