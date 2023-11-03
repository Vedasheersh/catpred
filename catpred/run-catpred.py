import pandas as pd
import argparse
import ete3
import numpy as np
import torch
import os
import json
import sys
from itertools import islice
from torch.utils import data as torch_data
from torchdrug import data, core
from torchdrug import utils as torchdrug_utils
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append('./src/')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import featurize
from model import CatPred
from dataset import CatPredDataset, CatPredSeqDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input csv file", type=str, required=True)
    parser.add_argument("-par", "--parameter", help="parameter to predict", 
                        type=str, required=True)

    args, unparsed = parser.parse_known_args()
    parser = argparse.ArgumentParser()

    return args

args = parse_args()

print(args)
root_path = os.path.abspath('.')
print(root_path)
data_dir = './data/'
model_dir = f'./models/{args.parameter.upper()}'
model_dicts = [torch.load(model_dir+'/'+file) for file in os.listdir(model_dir) if file.endswith('.pth')]
cfg = json.load(open(f'{model_dir}/config.json'))

dfin = pd.read_csv(args.input)
# df = dfin.copy()
# X, df = featurize(df, args.parameter.upper(), root_path, data_dir,
#                  redo_feats=True)


model = core.Configurable.load_config_dict(cfg["model"])

cfg["dataset"]["data_dir"] = data_dir
cfg["dataset"]["target"] = args.parameter.upper()
dataset = core.Configurable.load_config_dict(cfg["dataset"])
dataset.load_data(dfin)

embedder_model = core.Configurable.load_config_dict(cfg["embedder_model"])
embedder_model._load(dataset)
    
task = CatPred(model, embedder_model, task=dataset.tasks).to(device)
task.eval()

for model_dict in model_dicts:
    for key in ['mean','std','weight']:
        _ = model_dict.pop(key)

def get_prediction(task, batch):
    pred = task.predict(batch).item()
    return np.power(10, pred)

print('Making predictions ...')
preds_all = {}
dataloader = data.DataLoader(dataset, 1, num_workers=0)

for batch_id, batch in tqdm(enumerate(dataloader)):
    temp = []
    if task.device.type == "cuda":
        batch = torchdrug_utils.cuda(batch, device=task.device)        
    for model_dict in model_dicts:
        task.load_state_dict(model_dict)
        temp.append(get_prediction(task, batch))
    preds_all[batch['index'].item()] = temp

output_col_avg = []
output_col_std = []
for i, row in dataset.data.iterrows():
    output_col_avg.append(np.average(preds_all[i]))
    output_col_std.append(np.std(preds_all[i]))
    
if args.parameter.upper()=='KCAT':
    outname = 'KCAT s^(-1)'
elif args.parameter.upper()=='KM':
    outname = 'KM mM'
if args.parameter.upper()=='KI':
    outname = 'KI mM'

dfin[outname + ' - avg'] = output_col_avg
dfin[outname + ' - std'] = output_col_std
dfin.drop(columns=[col for col in dfin.columns if 'INTEGER' in col or \
                  'VEC' in col or 'ONEHOT' in col], inplace=True)

dfin.to_csv(f'{args.input[:-4]}_result.csv')