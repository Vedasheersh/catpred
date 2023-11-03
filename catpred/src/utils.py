import pandas as pd
import numpy as np
import time
import json
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import ete3
import torch

def load_vocabulary(parameter, root_path, data_dir):
    vocab_dic = json.load(open(f"{root_path}/{data_dir}/vocab/{parameter}_vocab.json"))
    return vocab_dic

def add_integer_embedding(vocab_dic, df, colname):
    dic = vocab_dic[colname]
    temp = []
    temp_vec = []
    for name in df[colname].astype("str"):
        if not name in dic: 
            name = 'UNK'
        temp.append(dic[name])
        temp_vec.append([dic[name]])
    
    df[f"{colname}_INTEGER"] = temp
    df[f"{colname}_INTEGER_VEC"] = temp_vec
    return df

def add_onehot_embedding(vocab_dic, df, colname):
    ints = df[f"{colname}_INTEGER"]
    dic = vocab_dic[colname]
    keys = []
    values = []
    for k, v in dic.items():
        values.append(int(v))
        keys.append(str(k))

    x = np.array(values).astype('int')
    onehot = np.zeros((len(df), len(x)))
    i = 0
    for each in ints:
        loc = values.index(each)
        onehot[i, loc] = 1
        i+=1

    onehot = list(onehot.astype('int'))
    df[f'{colname}_ONEHOT'] = onehot
    return df

def add_fps(df, radius=3, length=2048):
    failed = []
    fps = []
    for i, row in tqdm(df.iterrows()):
        smi = row["SMILES"]
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius,length)
            fps.append(np.array(fp))
        except:
            failed.append(i)
            fps.append(None)
            
    df["FP"] = fps
    df.drop(failed, inplace=True)
    return df
    
def get_ec_words(ec):
    if '-' in ec: ec.replace('-','UNK')
    ec_chars = ec.split('.')
    ec_words = {f"EC{i}": '.'.join(ec_chars[:i]) for i in range(1,4)}
    ec_words['EC'] = ec
    return ec_words

def get_tax_words(organism, ncbi, org_to_taxid, tax_embed_cols):
    get_taxid_from_organism = lambda organism: ncbi.get_name_translator([organism])[organism][0]
    try:
        taxid = get_taxid_from_organism(organism)
    except:
        if not organism in org_to_taxid:
            taxid = None
            print(f'Organism {organism} not found in NCBI or CatPred database! Making predictions using UNK words, this may lead to inaccurate predictions')
        else:
            taxid = org_to_taxid[organism]
            
        return {tax: 'UNK' for tax in tax_embed_cols}
    
    lineage = ncbi.get_lineage(taxid)
    rank_dict = ncbi.get_rank(lineage)
    rank_dict_return = {}
    for rankid, rankname in rank_dict.items():
        if rankname.upper() in tax_embed_cols: rank_dict_return[rankname.upper()] = ncbi.get_taxid_translator([rankid])[rankid]
        
    return rank_dict_return

def get_esm_features(df_w_seqs, esm_install_dir='./esm/', rep_type='mean'):
    unique_seqs = list(df_w_seqs.SEQUENCE.unique())
    unique_seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}
    
    df_w_seqs.reset_index(inplace=True, drop=True)
    
    dfid_to_seqid = {}
    for i, row in df_w_seqs.iterrows():
        seq = row.SEQUENCE
        dfid_to_seqid[i] = unique_seq_to_id[seq]
        
    f = open(f'{esm_install_dir}/temp_seqs.fasta','w')
    for seq, i in unique_seq_to_id.items():
        f.write(f'>{i}\n{seq}\n')
    f.close()
    
    if os.path.exists(f'{esm_install_dir}/temp_seqs_esm2'):
        os.system(f'rm {esm_install_dir}/temp_seqs_esm2/*')
    
    os.system(f'python {esm_install_dir}/scripts/extract.py esm2_t33_650M_UR50D {esm_install_dir}/temp_seqs.fasta {esm_install_dir}/temp_seqs_esm2 --repr_layers 33 --include {rep_type}')
    
    key = 'mean_representations' if rep_type=='mean' else 'representations'
    embed_dic = {i: torch.load(f'{esm_install_dir}/temp_seqs_esm2/{dfid_to_seqid[i]}.pt')[key][33].cpu().numpy() for i in df_w_seqs.index}
    
    if rep_type=='mean':
        return embed_dic
    else:
        # do something
        return embed_dic
    
def add_esm_features(df_w_seqs, esm_install_dir='./esm/', rep_type='mean'):
    embed_dic = get_esm_features(df_w_seqs, esm_install_dir='./esm/', rep_type='mean')
    df_w_seqs['ESM'] = [embed_dic[i] for i in df_w_seqs.index]
    return df_w_seqs

# def pool

def featurize(df, parameter, 
              root_path = ".", 
              data_dir = './data/', 
              redo_feats = False,
              baseline=False,
              add_esm=False,
              skip_embeds=False,
              skip_fp = False,
              fp_radius = 3, 
              fp_length = 2048,
              include_y = False):
    ec_embed_cols = ["EC1", "EC2", "EC3", "EC"]
    tax_embed_cols = [
        "SUPERKINGDOM",
        "PHYLUM",
        "CLASS",
        "ORDER",
        "FAMILY",
        "GENUS",
        "SPECIES",
    ]
    if redo_feats:
        print('Redoing feats...')
        ncbi = ete3.NCBITaxa(taxdump_file=f'{root_path}/{data_dir}/taxdump.tar.gz', update=False)
        org_to_taxid = json.load(open(f'{root_path}/{data_dir}/organism_to_taxid.json'))

        if add_esm: 
            assert('SEQUENCE' in df)
            print('Adding ESM2 features ...')
            df = add_esm_features(df)

        print("Preparing Data ...")

        print("Adding EC and TC words from pre-defined vocabulary ...")
        ec_words = []
        for ind, row in df.iterrows():
            words = get_ec_words(row.EC)
            ec_words.append(words)
        for col in ec_embed_cols:
            col_values = [ec_words[i][col] for i in range(len(df))]
            df[col] = col_values

        tax_words = []
        for ind, row in df.iterrows():
            words = get_tax_words(row.Organism, ncbi, org_to_taxid, tax_embed_cols)
            tax_words.append(words)
        for col in tax_embed_cols:
            col_values = []
            for i in range(len(df)):
                if col in tax_words[i]:
                    col_values.append(tax_words[i][col])
                else:
                    col_values.append('UNK')
            df[col] = col_values

        vocab_dic = load_vocabulary(parameter, root_path, data_dir)

        for EC in ec_embed_cols:
            df = add_integer_embedding(vocab_dic, df, EC)
        for TAX in tax_embed_cols:
            df = add_integer_embedding(vocab_dic, df, TAX)
            #add onehots
        for EC in ec_embed_cols:
            df = add_onehot_embedding(vocab_dic, df, EC)
        for TAX in tax_embed_cols:
            df = add_onehot_embedding(vocab_dic, df, TAX)

        print("Adding substrate fingerprints ...")
        df = add_fps(df, fp_radius, fp_length)
    
    features_to_add = []
    if not skip_fp:
        features_to_add.append(df[['FP']])
        
    if add_esm: features_to_add.append(df[['ESM']])
    
    if not skip_embeds:
        embed_type = 'ONEHOT'
        if baseline: start=-1
        else: start=0
        for each in ec_embed_cols[start:]+tax_embed_cols[start:]:
            features_to_add.append(df[[f"{each}_{embed_type}"]])

    # total minus default ones
    n_feats = len(features_to_add)

    prepared_df = pd.concat(features_to_add, axis=1)
    
    print(prepared_df.columns)

    X_vals = prepared_df.iloc[:,:].values # only feats
    
    Xs = []
    for i in range(n_feats):
        Xs.append(np.stack(X_vals[:,i]))
            
    X = np.concatenate(Xs, axis=-1)

    if include_y: 
        try:
            y = np.array(df[f'target_{parameter}'].values)
            return (X,y), df
        except:
            print(f'include_y option is true but, not column target_{parameter} was not found!')
            print('Exiting..')
            sys.exit(0)
            return X, df
    else: 
        return X, df