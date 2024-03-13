import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import pickle as pkl                 
import networkx as nx
import time
from tqdm import tqdm
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.task_dir = task_dir = args.data_path
        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1
        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.filters = defaultdict(lambda:set())
        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
        self.all_triple = np.concatenate([np.array(self.fact_triple), np.array(self.train_triple)], axis=0)

        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)
        self.idd_data = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)
            
        self.shuffle_train()
        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data)
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])
            
        if mode == 'train':
            self.len = len(self.train_data)
        elif mode == 'valid':
            self.len = len(self.valid_q)
        else:
            self.len = len(self.test_q)
                
    def addSampler(self, sampler):
        self.sampler = sampler
        self.getOneSubgraph = self.sampler.getOneSubgraph
        self.getBatchSubgraph = self.sampler.getBatchSubgraph
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # indexing
        if self.mode == 'train':
            sub, rel, obj = self.train_data[idx]
            sub = torch.LongTensor([sub]).unsqueeze(0)
            rel = torch.LongTensor([rel]).unsqueeze(0)
            obj = torch.LongTensor([obj]).unsqueeze(0)
        else:
            if self.mode == 'valid':
                query, answer = self.valid_q, self.valid_a
            elif self.mode == 'test':
                query, answer = self.test_q, self.test_a
            sub, rel = query[idx]
            sub, rel = torch.LongTensor([sub]), torch.LongTensor([rel])
            obj = torch.zeros((self.n_ent)).long()
            obj[answer[idx]] = 1
                    
        # subgraph sampling
        subgraph = self.getOneSubgraph(int(sub))
        return sub, rel, obj, subgraph
        
    def collate_fn(self, data):
        subs = torch.stack([_[0] for _ in data], dim=0)
        rels = torch.stack([_[1] for _ in data], dim=0)
        objs = torch.stack([_[2] for _ in data], dim=0)
        subgraph_list = [_[3] for _ in data]
        batch_subgraph = self.getBatchSubgraph(subgraph_list)
        
        # NOTE: we can not return sparse tensor here
        # thus, we return its indices and values which are dense tensor.
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_subgraph
        return subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return list(triples) + new_triples

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers
    
    def shuffle_train(self):
        # print('shuffle training set...')
        all_triple = self.all_triple
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]
        bar = int(n_all * self.args.fact_ratio)
        self.fact_data = np.array(self.double_triple(all_triple[:bar].tolist()))
        self.train_data = np.array(self.double_triple(all_triple[bar:].tolist()))
        
        if self.args.remove_1hop_edges:
            print('==> removing 1-hop links...')
            tmp_index = np.ones((self.n_ent, self.n_ent))
            tmp_index[self.train_data[:, 0], self.train_data[:, 2]] = 0
            save_facts = tmp_index[self.fact_data[:, 0], self.fact_data[:, 2]].astype(bool)
            self.fact_data = self.fact_data[save_facts]
            print('==> done')

        # update
        self.n_train = len(self.train_data)
        self.len = len(self.train_data)        
        
        # shuffle training data
        n_all = len(self.train_data)
        rand_idx = np.random.permutation(n_all)
        self.train_data = self.train_data[rand_idx]