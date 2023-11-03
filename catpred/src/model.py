import os
import torch
import math
import logging
import warnings

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from torch import nn
from torch.nn import functional as F

from torchdrug import core, utils, models, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from memory_efficient_attention_pytorch import Attention

from transformers import AutoModelWithLMHead, AutoTokenizer

logger = logging.getLogger()
RDLogger.DisableLog("rdApp.*")

@R.register("models.RdkitFingerprint")
class RdkitFingerprint(torch.nn.Module, core.Configurable):
    """
    Model Class to compute Rdkit fingerprint of molecule constructed fromn SMILES 
    No trainable parameters!
    Parameters:
        index_to_smiles (dic) : dictionary of index to smiles from dataset
        length (int, optional): length of fingerprint array
        radius (int, optional): radius to use for constructing fingerprint
    """

    def __init__(self, length=2048, radius=3):
        super(RdkitFingerprint, self).__init__()
        self.length=length
        self.radius=radius
        self.output_dim = length
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, graph, *args, **kwargs):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Molecule): :math:`n` molecule graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``graph_feature`` field:
                graph representations of shape :math:`(1, length)`
        """

        smi_list = graph.to_smiles(canonical=True)
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 
                                                    self.radius, 
                                                    self.length) for smi in smi_list]
        fps = np.array(fps)
        fps = torch.tensor(fps).to(self.device)
        return {"graph_feature": fps, "node_feature": None}

@R.register("models.ChemBERTa")
class ChemBERTa(torch.nn.Module, core.Configurable):
    """
    Model Class to compute Rdkit fingerprint of molecule constructed fromn SMILES 
    No trainable parameters!
    Parameters:
        index_to_smiles (dic) : dictionary of index to smiles from dataset
        length (int, optional): length of fingerprint array
        radius (int, optional): radius to use for constructing fingerprint
    """

    def __init__(self, max_length, model_uri, freeze=True):
        super(ChemBERTa, self).__init__()
        model = AutoModelWithLMHead.from_pretrained(model_uri)
        self.model = model.roberta.to('cuda')
       
        if not freeze:
            self.model.train()
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_uri,use_fast=True)
        
        self.output_dim = 384
        self.max_length = max_length
        
    def forward(self, graph, *args, **kwargs):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Molecule): :math:`n` molecule graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``graph_feature`` field:
                graph representations of shape :math:`(1, length)`
        """

        smi_list = graph.to_smiles(canonical=True)
        tokens = self.tokenizer(smi_list, 
                                return_tensors='pt', 
                                max_length=self.max_length, 
                                padding='max_length',
                                return_attention_mask=True,
                                truncation=True).to(self.device)
        
        for k,v in tokens.items():
            tokens[k] = v.to(self.device)
            
        logits = self.model(**tokens)
        feats = logits['last_hidden_state'].to(self.device)
            
        return {"graph_feature": feats.mean(dim=1), 
                "node_feature": feats}
    
@R.register("models.ProtBert")
class ProtBert(torch.nn.Module, core.Configurable):
    """
    Model Class to compute Rdkit fingerprint of molecule constructed fromn SMILES 
    No trainable parameters!
    Parameters:
        index_to_smiles (dic) : dictionary of index to smiles from dataset
        length (int, optional): length of fingerprint array
        radius (int, optional): radius to use for constructing fingerprint
    """

    def __init__(self, max_length=1024, model_uri="Rostlab/prot_bert", freeze=True):
        super(ProtBert, self).__init__()
        self.model = AutoModelWithLMHead.from_pretrained(model_uri).bert.to('cuda')
       
        if not freeze:
            self.model.train()
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_uri,use_fast=True)
        
        self.output_dim = 1024
        self.max_length = max_length
        
    def forward(self, graph):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Molecule): :math:`n` molecule graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``graph_feature`` field:
                graph representations of shape :math:`(1, length)`
        """

        seq_list = graph.to_sequence()
        seq_list = [' '.join(list(s)) for s in seq_list]
        tokens = self.tokenizer(seq_list, 
                                return_tensors='pt', 
                                max_length=self.max_length, 
                                padding='max_length',
                                return_attention_mask=True,
                                truncation=True)
        
        for k,v in tokens.items():
            tokens[k] = v.to('cuda')

        logits = self.model(**tokens)
        feats = logits['last_hidden_state']
            
        return {"graph_feature": feats.mean(dim=1), 
                "node_feature": feats}, tokens["attention_mask"]

@R.register("models.ESM2")
class ESM2(torch.nn.Module, core.Configurable):
    """
    Model Class to compute Rdkit fingerprint of molecule constructed fromn SMILES 
    No trainable parameters!
    Parameters:
        index_to_smiles (dic) : dictionary of index to smiles from dataset
        length (int, optional): length of fingerprint array
        radius (int, optional): radius to use for constructing fingerprint
    """

    def __init__(self, max_length=1024, model_uri="Rostlab/prot_t5_xl_uniref50", freeze=True):
        super(ESM2, self).__init__()
        self.model = AutoModelWithLMHead.from_pretrained(model_uri).esm.to('cuda')
       
        if not freeze:
            self.model.train()
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_uri,use_fast=True)
        
        self.output_dim = 1280
        self.max_length = max_length
        
    def forward(self, graph):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Molecule): :math:`n` molecule graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``graph_feature`` field:
                graph representations of shape :math:`(1, length)`
        """

        seq_list = graph.to_sequence()
        tokens = self.tokenizer(seq_list, 
                                return_tensors='pt', 
                                max_length=self.max_length, 
                                padding='max_length',
                                return_attention_mask=True,
                                truncation=True).to('cuda')

        for k,v in tokens.items():
            tokens[k] = v.to('cuda')
            
        logits = self.model(**tokens)
        feats = logits['last_hidden_state']
            
        return {"graph_feature": feats.mean(dim=1), 
                "node_feature": feats}, tokens["attention_mask"]

@R.register("models.SerialEmbedder")
class SerialEmbedder(nn.Module):
    def __init__(self, embed_sizes, embed_dims):

        super().__init__()
        self.embed_sizes = embed_sizes
        self.embed_dims = embed_dims
        
        self.embed_layers = nn.ModuleList([])
        for size, dim in zip(self.embed_sizes, self.embed_dims):
            self.embed_layers.append(nn.Linear(size, dim, bias=False))

    def forward(self, x):
        outs = []
        for i, e in enumerate(x):
            layer = self.embed_layers[i]
            # print(e.dtype)
            out = layer(e.T)
            outs.append(out)

        outs = torch.stack(outs)
        
        return outs

@R.register("models.TaxEcEmbedder")
class TaxEcEmbedder(nn.Module, core.Configurable):
    def __init__(
        self,
        embed_dim,
        embed_hidden_dim,
        skip_EC=False,
        skip_tax=False,
        use_attn=True,
        baseline=False
    ):

        super().__init__()
        self.embed_dim = embed_dim
        self.embed_hidden_dim = embed_hidden_dim
        self.skip_EC = skip_EC
        self.skip_tax = skip_tax
        self.baseline = baseline
        self.use_attn = use_attn
    
    def _load(self, dataset):
        def _get_embedding_sizes(df, colname):
            embsize = len(df[colname + "_ONEHOT"].iloc[0])
            return embsize, self.embed_dim
        
        df = dataset.data
        ec_embed_sizes = []
        ec_embed_dims = []
        tax_embed_sizes = []
        tax_embed_dims = []

        if self.baseline: n=-1
        else: n=0
        
        for ec in dataset.ec_embed_cols[n:]:
            emb_size, emb_dim = _get_embedding_sizes(df, ec)
            ec_embed_sizes.append(emb_size)
            ec_embed_dims.append(emb_dim)
            
        for tax in dataset.tax_embed_cols[n:]:
            emb_size, emb_dim = _get_embedding_sizes(df, tax)
            tax_embed_sizes.append(emb_size)
            tax_embed_dims.append(emb_dim)

        self.ec_embed_sizes = ec_embed_sizes
        self.ec_embed_dims = ec_embed_dims

        self.tax_embed_sizes = tax_embed_sizes
        self.tax_embed_dims = tax_embed_dims

        self.ec_embedder = SerialEmbedder(
            self.ec_embed_sizes, self.ec_embed_dims)
        
        self.tax_embedder = SerialEmbedder(
            self.tax_embed_sizes, self.tax_embed_dims)

        if self.skip_EC:
            self.outlayer = nn.Linear(
                sum(self.tax_embed_dims), 
                self.embed_hidden_dim
            )
        elif self.skip_tax:
            self.outlayer = nn.Linear(
                sum(self.ec_embed_dims), 
                self.embed_hidden_dim
            )
        else:
            self.outlayer = nn.Linear(
                sum(self.ec_embed_dims) + 
                sum(self.tax_embed_dims), 
                self.embed_hidden_dim
            )

        if self.use_attn:
            self.attn_layer = Attention(
                        dim = emb_dim,
                        dim_head = 64,              
                        heads = 8,                   
                        causal = True,                
                        memory_efficient = True,      
                        q_bucket_size = 1,         
                        k_bucket_size = 1          
                    ).to(self.device) 
        
        self.output_dim = self.embed_hidden_dim

    def forward(self, batch, all_loss=None, metric=None):
        ecs = []
        taxs = []
        # print(type(batch["index"]))
        if type(batch["index"]) is int: #for predict_and_target
            for each in batch["ec_vector_onehot"]:
                tens = torch.tensor(each).to(self.device).unsqueeze(dim=-1)
                ecs.append(tens.to(torch.float32))
            for each in batch["tax_vector_onehot"]:
                tens = torch.tensor(each).to(self.device).unsqueeze(dim=-1)
                taxs.append(tens.to(torch.float32))
        elif type(batch["index"]) is torch.Tensor:   
            for each in batch["ec_vector_onehot"]:
                tens = torch.stack(each)
                ecs.append(tens)
            for each in batch["tax_vector_onehot"]:
                tens = torch.stack(each)
                taxs.append(tens)

        ec_outs = self.ec_embedder(ecs)
        tax_outs = self.tax_embedder(taxs)
                
        ec_outs = torch.transpose(ec_outs, 0,1)
        tax_outs = torch.transpose(tax_outs, 0,1)
        
        if self.skip_EC:
            outs = tax_outs
        elif self.skip_tax:
            outs = ec_outs
        elif self.skip_tax and self.skip_EC:
            return None, None
        else:
            cats = torch.cat([ec_outs, tax_outs], dim=1)
                
        if self.use_attn:
            # print(cats.shape)
            embeds = torch.reshape(cats, (cats.shape[0],11,self.embed_dim))
            # print(embeds.shape)
            embeds_attn = self.attn_layer(embeds)
            embeds_out = embeds_attn.flatten(1,2)
            # print(embeds_out.shape)
            outs = F.relu(self.outlayer(embeds_out))

            return outs, embeds_attn

        else:
            outs = F.relu(self.outlayer(cats))
            
        return outs, None
    
@utils.copy_args(tasks.PropertyPrediction)
class CatPred(tasks.PropertyPrediction):
    """
    Predict the interaction property of graph pairs.

    Parameters:
        substrate_model (nn.Module): graph representation model
        embedder_model (nn.Module, optional): TaxEcEmbedder model
        only_substrate (bool, optional): Use only substrate features or not
        skip_substrate (bool, optional): Skip substrate features or not
        return_embeds (bool, optional): XX
        skip_attention (bool, optional): XX
        **kwargs
    """

    def __init__(
        self,
        substrate_model,
        embedder_model,
        only_substrate=False,
        skip_substrate=False,
        return_embeds=False,
        skip_attention=False,
        **kwargs
    ):
        super(CatPred, self).__init__(substrate_model, **kwargs)
        self.substrate_model = substrate_model
        self.embedder_model = embedder_model
        self.only_substrate = only_substrate
        self.skip_substrate = skip_substrate
        self.return_embeds = return_embeds
        self.skip_attention = skip_attention
        
        self.num_class = [1]

        if self.skip_substrate:
            mlp_in_dim = self.embedder_model.output_dim
            mlp_outs = [256, 64] + [sum(self.num_class)]
            
        elif self.only_substrate:
            mlp_in_dim = self.substrate_model.output_dim
            mlp_outs = [1024, 256, 64] + [sum(self.num_class)]
            
        else:
            mlp_in_dim = self.substrate_model.output_dim + self.embedder_model.output_dim
            mlp_outs = [1024, 256, 64] + [sum(self.num_class)]

        self.mlp = layers.MLP(mlp_in_dim, mlp_outs, batch_norm=True)

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        
    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]

        outs_graph = self.substrate_model(graph, graph.node_feature.float(), 
                        all_loss=all_loss, metric=metric)
        
        if not self.only_substrate:
            enz_outs, embeds_attn = self.embedder_model(batch, all_loss, metric)

        if self.skip_substrate:
            cat_feats = enz_outs
            
        elif self.only_substrate:
            cat_feats = outs_graph["graph_feature"]
            
        else:
            cat_feats = torch.cat([outs_graph["graph_feature"], enz_outs], dim=-1)

        pred = self.mlp(cat_feats, self.return_embeds)

        return pred
    
@utils.copy_args(tasks.PropertyPrediction)
class CatPredSeq(tasks.PropertyPrediction):
    """
    Predict the interaction property of graph pairs.

    Parameters:
        substrate_model (nn.Module): graph representation model
        enzyme_model (nn.Module): graph representation model
        embedder_model (nn.Module, optional): TaxEcEmbedder model
        only_substrate (bool, optional): Use only substrate features or not
        skip_substrate (bool, optional): Skip substrate features or not
        return_embeds (bool, optional): XX
        skip_attention (bool, optional): XX
        **kwargs
    """

    def __init__(
        self,
        substrate_model,
        enzyme_model,
        embedder_model,
        only_substrate=False,
        skip_substrate=False,
        return_embeds=False,
        skip_attention=False,
        **kwargs
    ):
        super(DeepCatPredSeq, self).__init__(substrate_model, **kwargs)
        self.substrate_model = substrate_model
        self.enzyme_model = enzyme_model
        self.embedder_model = embedder_model
        self.only_substrate = only_substrate
        self.skip_substrate = skip_substrate
        self.return_embeds = return_embeds
        self.skip_attention = skip_attention

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        self.attn = BidirectionalCrossAttention(
            dim = self.enzyme_model.output_dim,
            heads = 8,
            dim_head = 64,
            context_dim = self.substrate_model.output_dim
        )
        
        mlp_in_dim = self.substrate_model.output_dim + (
            len(self.embedder_model.ec_embed_dims) +
            len(self.embedder_model.tax_embed_dims)
        ) * self.embedder_model.embed_dim + self.enzyme_model.output_dim
        
        mlp_outs = [1024, 256, 64] + [sum(self.num_class)]

        self.mlp = layers.MLP(mlp_in_dim, mlp_outs)
        
    def predict(self, batch, all_loss=None, metric=None):

        # print(batch.keys())
        sub_logits, sub_mask = self.substrate_model(batch["substrate_graph"])
        sub_mask = sub_mask.bool()
        sub_logits = sub_logits["node_feature"]

        emb_outs, _ = self.embedder_model(batch, all_loss, metric)
        enz_logits, enz_mask = self.enzyme_model(batch["enzyme_graph"])
        enz_logits = enz_logits["node_feature"]
        enz_mask = enz_mask.bool()
        
        zeros = torch.zeros(sub_logits.shape[0], 
                            self.substrate_model.max_length, 
                            sub_logits.shape[-1]
                            ).to(self.device)
        zeros[:,:sub_logits.shape[1],:] = sub_logits
        sub_outs = zeros
        
        zeros = torch.zeros(enz_logits.shape[0], 
                            self.enzyme_model.max_length, 
                            enz_logits.shape[-1]
                            ).to(self.device)
        zeros[:,:enz_logits.shape[1],:] = enz_logits
        enz_outs = zeros

        # print(enz_outs.shape, enz_mask.shape, sub_outs.shape, sub_mask.shape)
        enz_outs, sub_outs = self.attn(enz_outs, sub_outs, 
                                       mask = enz_mask, 
                                       context_mask = sub_mask)

        # print(enz_outs.shape, sub_outs.shape)
        
        enz_outs = torch.mean(enz_outs, dim=1)
        sub_outs = torch.mean(sub_outs, dim=1)
        emb_outs = torch.flatten(emb_outs, start_dim=1)
        
        # print(enz_outs.shape, sub_outs.shape, emb_outs.shape)
        
        cat_feats = torch.cat([enz_outs, sub_outs, emb_outs],dim=-1)

        # print(cat_feats.shape)
        
        pred = self.mlp(cat_feats, self.return_embeds)

        if self.normalization:
            pred = pred * self.std + self.mean

        return pred