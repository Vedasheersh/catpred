import os
import torch
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import RDLogger

from tqdm import tqdm
from joblib import Parallel, delayed

from torch.utils import data as torch_data
from torch.utils.data import Dataset

from torchdrug.core import Registry as R
from torchdrug import data, datasets
import ete3
import json

RDLogger.DisableLog("rdApp.*")
os.environ["TORCH_VERSION"] = torch.__version__

def _get_pdb_graph(pdb_file):
    try:
        graph = data.Protein.from_pdb(pdb_file)
    except:
        graph = None
    return graph

def _get_seq_graph(seq):
    try:
        graph = data.Protein.from_sequence(seq)
    except:
        graph = None
    return graph

def load_vocabulary(parameter, data_dir):
    vocab_dic = json.load(open(f"{data_dir}/vocab/{parameter}_vocab.json"))
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

    x = np.array(values)
    onehot = np.zeros((len(df), len(x)))
    i = 0
    for each in ints:
        loc = values.index(each)
        onehot[i, loc] = 1
        i+=1

    onehot = list(onehot)
    df[f'{colname}_ONEHOT'] = onehot
    return df

def add_fps(df, radius=2, length=2048):
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

def _featurize(df, data_dir, parameter, ncbi, org_to_taxid):
    
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

    vocab_dic = load_vocabulary(parameter, data_dir)

    for EC in ec_embed_cols:
        df = add_integer_embedding(vocab_dic, df, EC)
    for TAX in tax_embed_cols:
        df = add_integer_embedding(vocab_dic, df, TAX)
        #add onehots
    for EC in ec_embed_cols:
        df = add_onehot_embedding(vocab_dic, df, EC)
    for TAX in tax_embed_cols:
        df = add_onehot_embedding(vocab_dic, df, TAX)
    
    return df
   
@R.register("datasets.CatPredDataset")
class CatPredDataset(data.MoleculeDataset):
    """

    Statistics:
        - #Molecule:
        - #Regression task: 1

    Parameters:
        dataframe (pandas.DataFrame): dataframe object
        target (str): target kinetic parameter (KCAT or KM or KI)
        verbose (int, optional): output verbose level
        in_memory (bool, optional): if store data in memory or not
        target (str, optional): column of target in data csv
        pretrained_graph_feats (bool, optional): load pretrained features for substrate graphs or not
        **kwargs
    """

    splits = ["test", "train", "valid"]

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

    substrate_col = "SMILES"
    split_col = "split"

    def __init__(
        self,
        target,
        verbose=1,
        in_memory=True,
        data_dir = './data',
        **kwargs,
    ):
        self.target_fields = [f"target_{target}"]
        self.target = target
        self.in_memory = in_memory
        self.data_dir = data_dir

    def featurize(self):
        # remove entries not Graph-able
        failed = []
        df = self.data
        for ind, row in df.iterrows():
            smiles = row[self.substrate_col]
            try:
                mol = Chem.MolFromSmiles(smiles)
                graph = data.Molecule.from_molecule(mol)
                smi = graph.to_smiles()
                mol = graph.to_molecule(ignore_error=True)
                if mol is None:
                    failed.append(ind)
                    continue
            except:
                failed.append(ind)
                continue

        print(f'Removed {len(failed)} entries because their SMILES failed to generate a valid Graph and back')
        df.drop(failed, inplace=True)
        df.reset_index(inplace=True, drop=True)

        ncbi = ete3.NCBITaxa(taxdump_file=f'{self.data_dir}/taxdump.tar.gz', update=False)
        org_to_taxid = json.load(open(f'{self.data_dir}/organism_to_taxid.json'))
    
        df = _featurize(df, self.data_dir, self.target, ncbi, org_to_taxid)

        self.data = df.reset_index(drop=True)

    def load_data(self, dataframe):
        dataframe.reset_index(inplace=True,drop=True)
        self.data = dataframe
        self.featurize()
        
        self.loaded = False
        if self.in_memory:
            self.in_memory_data = []
            for ind, row in tqdm(self.data.iterrows()):
                self.in_memory_data.append(self.get_item(ind))
            self.loaded = True

    @property
    def tasks(self):
        """List of tasks."""
        return self.target_fields

    def split(self, keys=[]):
        splits = []
        if not keys:
            keys = self.splits
        for split_name in keys:
            split = torch_data.Subset(self, self.split_indices[split_name])
            splits.append(split)
        return splits

    def get_item(self, index):

        if self.in_memory and self.loaded:
            return self.in_memory_data[index]

        row = self.data.loc[index]
        smiles = row[self.substrate_col]
            
        ec_vec = []
        for col in self.ec_embed_cols:
            ec_vec.append(int(row[col + "_INTEGER"]))

        tax_vec = []
        for col in self.tax_embed_cols:
            tax_vec.append(int(row[col + "_INTEGER"]))
            
        ec_vec_onehot = []
        for col in self.ec_embed_cols:
            ec_vec_onehot.append(list(row[col + "_ONEHOT"]))

        tax_vec_onehot = []
        for col in self.tax_embed_cols:
            tax_vec_onehot.append(list(row[col + "_ONEHOT"]))
        
        ec_onehot = list(row['EC_ONEHOT'])
        sp_onehot = list(row['SPECIES_ONEHOT'])
                
        mol = Chem.MolFromSmiles(smiles)
        graph = data.Molecule.from_molecule(mol)

        ec_return = ec_vec_onehot
        tax_return = tax_vec_onehot
            
        returner = {
            "graph": graph,
            "ec_vector_onehot": ec_return,
            "tax_vector_onehot": tax_return,
            "ec1": [ec_vec[0]],
            "tax1": [tax_vec[0]],
            "ec_vector": ec_vec[1:],
            "tax_vector": tax_vec[1:],
            "index": index
        }
                
        return returner

@R.register("datasets.CatPredSeqDataset")
class CatPredSeqDataset(CatPredDataset):
    """
    Statistics:
        - #Molecule:
        - #Regression task: 1

    Parameters:
        dataframe (pandas.DataFrame): dataframe object
        target (str): target kinetic parameter (KCAT or KM or KI)
        pdb_path (str): path to saved pdb files
        verbose (int, optional): output verbose level
        in_memory (bool, optional): if store data in memory or not
        pretrained_graph_feats (bool, optional): load pretrained features for substrate graphs or not
        save_dir (str): path to directory to save preprocessed data
        **kwargs
    """

    def __init__(
        self,
        target,
        pdb_path,
        seq_only=True,
        save_dir=None,
        **kwargs,
    ):

        if not save_dir is None:
            save_dir = os.path.expanduser(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_dir = save_dir

        self.pdb_path = pdb_path
        self.seq_only = seq_only

        CatPredDataset.__init__(self, target, **kwargs)

    def load_data(self, dataframe):
        def _load(args):
            obj, ind = args
            return obj.get_item(ind)

        def _get_smiles_graph(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if self.pretrained:
                    graph2 = data.Molecule.from_molecule(
                        mol, node_feature="pretrain", 
                        edge_feature="pretrain"
                    )
                else:
                    graph2 = data.Molecule.from_molecule(mol)
            except:
                graph2 = None

            return graph2

        dataframe.reset_index(inplace=True,drop=True)
        self.data = dataframe
        self.featurize()
        split_indices = {}
        prev = 0
        for name in self.splits:
            now = len(self.data[self.data[self.split_col] == name])
            split_indices[name] = range(prev, now + prev)
            prev = now + prev

        self.split_indices = split_indices
        self.pdb_files = [
            os.path.join(self.pdb_path, uniprot + ".pdb")
            for uniprot in self.data["UNIPROT"]
        ]

        self.sequences = self.data["SEQUENCE"]
        self.smiles = self.data[self.substrate_col]

        self.loaded = False

        print("Creating protein graphs..")
        if self.seq_only:
            self.pdb_graphs = Parallel(n_jobs=30, verbose=1)(
            delayed(_get_seq_graph)(seq) for seq in self.sequences
        )
        else:
            self.pdb_graphs = Parallel(n_jobs=30, verbose=1)(
                delayed(_get_pdb_graph)(pdbfile) for pdbfile in self.pdb_files
            )

        print("Creating smiles graphs..")
        self.smiles_graphs = []
        for sm in tqdm(self.smiles):
            self.smiles_graphs.append(_get_smiles_graph(sm))

        self.loaded = True

    def get_item(self, index):
        row = self.data.iloc[index]
        smiles = row.SUBSTRATE_SMILES
        y = row[self.target_fields[0]]

        if self.in_memory:
            graph1 = self.pdb_graphs[index]
            graph2 = self.smiles_graphs[index]
        else:
            if self.seq_only:
                seq = self.sequences[index]
                graph1 = _get_seq_graph(seq)
            else:
                pdb_file = self.pdb_files[index]
                graph1 = _get_pdb_graph(pdb_file)

            mol = Chem.MolFromSmiles(smiles)
            if self.pretrained:
                graph2 = data.Molecule.from_molecule(
                    mol, node_feature="pretrain", 
                    edge_feature="pretrain"
                )
            else:
                graph2 = data.Molecule.from_molecule(mol)

        ec_vec_onehot = []
        for col in self.ec_embed_cols:
            ec_vec_onehot.append(list(row[col + "_ONEHOT"]))

        tax_vec_onehot = []
        for col in self.tax_embed_cols:
            tax_vec_onehot.append(list(row[col + "_ONEHOT"]))
        
        ec_onehot = list(row['EC_ONEHOT'])
        sp_onehot = list(row['SPECIES_ONEHOT'])
        
        return {
            self.target_fields[0]: y,
            "index": index,
            "ec_vector_onehot": ec_vec_onehot[:],
            "tax_vector_onehot": tax_vec_onehot[:],
            "enzyme_graph": graph1, 
            "substrate_graph": graph2
        }
