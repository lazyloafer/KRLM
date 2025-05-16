import os
import pickle
import random

import torch
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from configs import train_args
from tasks import *
# from utils import build_relation_graph

class KG:
    def __init__(self, args, ind=''):
        self.args = args
        self.ind = ind
        # self.triples = triples
        # self.n_rel = n_rel
        # self.n_ent = n_ent
        self.get_kg()
        self.get_markov_path_set()
        # self.build_rules(case_num=25, hop=3)

    def get_kg(self):
        with open(os.path.join(self.args.data_path, self.args.dataset, f'train{self.ind}.txt'), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(self.args.data_path, self.args.dataset, f'valid{self.ind}.txt'), 'r') as f:
            lines += f.readlines()
        if self.args.trans_or_ind == 'transductive' or self.ind == '_ind':
            with open(os.path.join(self.args.data_path, self.args.dataset, f'test{self.ind}.txt'), 'r') as f:
                lines += f.readlines()
        self.triples = []
        for line in lines:
            h, r, t = line.strip().split('\t')
            # if int(r) % 2 == 0:
            self.triples.append((int(h), int(r), int(t)))
        with open(os.path.join(self.args.data_path, self.args.dataset, f'stats{self.ind}.txt'), 'r') as f:
            stats = f.readlines()
        self.n_ent = int(stats[0].strip().split(': ')[-1])
        self.n_rel = int(stats[1].strip().split(': ')[-1])
        self.r2h = get_r2h(self.triples)
        self.r2t = get_r2t(self.triples)
        self.ht2r = get_ht2r(self.triples)
        self.rel2triples = get_rel2triples(self.triples)
        self.G = self.build_nx_graph()
        # self.n_rel = int(int(stats[1].strip().split(': ')[-1]) / 2)

    def markov_random_walk(self, G, start_node, max_steps=3):
        current_node = start_node
        path = []
        score = 0

        for i in range(max_steps):
            neighbors = list(G.out_edges(current_node, keys=True))
            if not neighbors:
                break
            path += [current_node]
            next_edge = random.choice(neighbors)
            next_node = next_edge[1]
            relation = next_edge[2] + self.n_ent
            # path.append((current_node, relation, next_node))
            path += [relation]
            in_degree = G.in_degree(current_node)
            out_degree = G.out_degree(next_node)

            score += math.exp(-i)/(math.sqrt(in_degree) * math.sqrt(out_degree))
            current_node = next_node
        path += [current_node]

        return (score / max_steps, path)

    def get_paths(self, e, max_steps=3, sampled_paths=100):
        # for e in tqdm(range(self.n_ent)):
        print(f'start entity {e}')
        paths = []
        for j in range(sampled_paths):
            p = self.markov_random_walk(self.G, e, max_steps)
            if p not in paths:
                paths.append(p)
        # for i in range(1, max_steps + 1):
        #     for j in range(sampled_paths):
        #         p = self.markov_random_walk(self.G, e, i)
        #         if p not in paths:
        #             paths.append(p)
        print(f'finish entity {e}')
        self.markov_path_set[e] = paths

    def get_markov_path_set(self):
        self.markov_path_set = {}
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     futures = [executor.submit(self.get_paths, e) for e in range(self.n_ent)]
        #     for future in tqdm(as_completed(futures)):  # tqdm(as_completed(futures), total=len(entities)):
        #         e, paths = future.result()
        #         self.markov_path_set[e] = paths
        with ThreadPoolExecutor(max_workers=20) as executor:
            for e in range(self.n_ent):
                executor.submit(self.get_paths, e)
        with open(os.path.join(self.args.data_path, self.args.dataset, f'markov_path_set{self.ind}.pkl'), 'wb') as f:
            pickle.dump(self.markov_path_set, f)

    def build_nx_graph(self):
        G = nx.MultiDiGraph()
        for triple in self.triples:
            G.add_edge(triple[0], triple[2], key=triple[1])
        return G

class Create_LQL(object):
    def __init__(self, args, cut_off, ind='', full=False):
        self.cut_off = cut_off
        self.ind = ind
        self.full = full
        self.sampling_num = 5
        self.args = args
        self.struct2type = {
            ("e", ("r",)): "1p",
            ("e", ("r", "r")): "2p",
            ("e", ("r", "r", "r")): "3p",
            (("e", ("r",)), ("e", ("r",))): "2i",
            (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
            ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
            (("e", ("r", "r")), ("e", ("r",))): "pi",
            (("e", ("r",)), ("e", ("r", "n"))): "2in",
            (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
            ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
            (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
            (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
            (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
            ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
            ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
        }

        # with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_test.pkl'), 'rb') as f:
        #     ss = pickle.load(f)
        #
        # for i in ss['data']['alternative_t_extra'].tolist():
        #     print(len(i))

        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(self.args.llm_path, self.args.llm_name),
        #                                                use_fast=False, add_eos_token=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='./llm_source/Llama-2-7b-chat-hf',
            use_fast=False, add_eos_token=False)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'right'
        self.org_vocab_size = self.tokenizer.vocab_size

        self.ent2id, self.rel2id, self.entid2name, self.name2entid, self.relid2name, self.name2relid = self.load_dict()
        self.num_ent, self.num_rel = len(self.ent2id), len(self.rel2id)
        # self.new_vocab_size = self.num_ent + self.num_rel
        # self.relation_paths = self.load_relation_paths()

        # self.inst = "### Instruction:\nSuppose you are a linguistic expert who are learning a new language. Given the following vocabulary:\n\n"
        # self.inst = "### Instruction:\nSuppose you are a linguistic expert who are learning a new language. Given the following case sentences:\n\n"
        self.inst = "### Instruction:\nDefine the word format for a new language as <Type: Text Description>. Suppose you are a linguistic expert who are learning this new language. Given the following vocabulary:\n\n"
        # self.vocab_placeholder = "Vocabulary:\nWord\tType\tText Description\n{Head}\t{Head_Type}\t{Head_Desc}\n{Rel}\t{Rel_Type}\t{Rel_Desc}\n\n"
        self.vocab_placeholder = "Vocabulary:\nWord\tType\tText Description\tGraph Representation\n{Head}\t{Head_Type}\t{Head_Desc}\t{Head_Graph}\n{Rel}\t{Rel_Type}\t{Rel_Desc}\t{Rel_Graph}\n\n"
        self.chain_placeholder = "Logical chains:\n{cases}\n"
        self.case_placeholder = "Case sentences:\n\n{cases}\n"
        self.query_placeholder = "Please complete the next word '?' in the given sentence:\n{Query}\n\n"
        self.response = "### Response:\n{Head}{Rel}"

        train_data = self.create_lql_from_kg('train')
        valid_data = self.create_lql_from_kg('valid')
        if self.args.trans_or_ind == 'transductive' or self.ind == '_ind':
            test_data = self.create_lql_from_kg('test')
        print('finish!')

    def load_dict(self):
        print('Loading entity information ......')
        with open(os.path.join(self.args.data_path, self.args.dataset, f'ent2id{self.ind}.pkl'), 'rb') as f:
            ent2id = pickle.load(f)
        with open(os.path.join(self.args.data_path, self.args.dataset, f'id2ent{self.ind}.pkl'), 'rb') as f:
            id2ent = pickle.load(f)

        entid2name = {}
        with open(os.path.join(self.args.data_path, self.args.dataset, 'name_entity.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            ent, name = line.strip().split('\t')
            if 'WN18RR' in self.args.dataset:
                name = name.split(',')[0]
            if ent in ent2id:
                entid2name[ent2id[ent]] = name
        entid2name = dict(sorted(entid2name.items(), key=lambda x: x[0]))
        name2entid = {v: k for k, v in entid2name.items()}

        print('Loading relation information ......')
        with open(os.path.join(self.args.data_path, self.args.dataset, f'rel2id{self.ind}.pkl'), 'rb') as f:
            rel2id = pickle.load(f)
        with open(os.path.join(self.args.data_path, self.args.dataset, f'id2rel{self.ind}.pkl'), 'rb') as f:
            id2rel = pickle.load(f)

        relid2name = {}
        with open(os.path.join(self.args.data_path, self.args.dataset, 'name_relation.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            rel, name = line.strip().split('\t')
            if f'+{rel}' in rel2id:
                relid2name[rel2id[f'+{rel}']] = name
                relid2name[rel2id[f'-{rel}']] = f'inverse of {name}'
        relid2name = dict(sorted(relid2name.items(), key=lambda x: x[0]))
        name2relid = {}
        repetitive_relname = {}
        for k, v in relid2name.items():
            if v not in name2relid:
                name2relid[v] = k
            else:
                if v not in repetitive_relname:
                    repetitive_relname[v] = 1
                else:
                    repetitive_relname[v] += 1
                relid2name[k] = f'{v} #{repetitive_relname[v]}'
                name2relid[f'{v} #{repetitive_relname[v]}'] = k


        # self.tokenizer.add_tokens(ent_tokens + rel_tokens)
        return ent2id, rel2id, entid2name, name2entid, relid2name, name2relid

    def load_relation_paths(self):
        # relation_paths = {}
        # for r in range(len(self.rel2id)):
        #     with open(os.path.join(self.args.data_path, self.args.dataset, 'relation_paths', f'{r}_paths.pkl'), 'rb') as f:
        #         r_paths = pickle.load(f)
        #         r_paths = sorted(r_paths, key=lambda x: x[0], reverse=True)
        #         relation_paths[r] = r_paths
        with open(os.path.join(self.args.data_path, self.args.dataset, f'markov_path_set{self.ind}.pkl'), 'rb') as f:
            relation_paths = pickle.load(f)
        for i in range(self.num_ent):
            paths = []
            weights = []
            for p in relation_paths[i]:
                weights.append(p[0])
                paths.append(p[1])
            relation_paths[i] = [weights, paths]
        return relation_paths

    def sampling_relation_paths(self, h, num=10):
        # paths = []
        # scores = []
        # for p in self.relation_paths[r]:
        #     scores.append(p[0])
        #     paths.append(p[1])
        weights, paths = self.relation_paths[h]
        sampled_paths = random.choices(paths, k=self.sampling_num, weights=list(np.array(weights) / sum(weights)))
        path_text = ''
        for i in range(len(sampled_paths)):
            path_text += f"Sentence {i+1}: "
            for k in range(len(sampled_paths[i])):
                if k % 2 == 0:
                    path_text += f'<entity: {self.entid2name[sampled_paths[i][k]]}>'
                else:
                    path_text += f'<relation: {self.relid2name[sampled_paths[i][k] - self.num_ent]}>'
            path_text += '\n\n'
        return path_text

    def get_text(self, h_lql, h, r_lql, r):
        vocab_param = {'Head': h_lql, 'Head_Type': 'Entity', 'Head_Desc': self.entid2name[h], 'Head_Graph': '<Ent_Graph>',
                       'Rel': r_lql, 'Rel_Type': 'Relation', 'Rel_Desc': self.relid2name[r], 'Rel_Graph': '<Rel_Graph>'}
        query_param = {'Query': f"{h_lql}{r_lql}?"}
        repseonse_param = {'Head': h_lql, 'Rel': r_lql}
        # case_param = {'cases': self.sampling_relation_paths(h)}
        text = f"{self.inst}{self.vocab_placeholder.format(**vocab_param)}{self.query_placeholder.format(**query_param)}{self.response.format(**repseonse_param)}"
        # text = f"{self.inst}{self.case_placeholder.format(**case_param)}{self.query_placeholder.format(**query_param)}{self.response.format(**repseonse_param)}"
        return text

    def create_lql_from_kg(self, mode):
        print(f'Create LQL from {mode}_kg ......')
        ent_tokens = [f"<Entity: {self.entid2name[i]}>" for i in range(len(self.entid2name))]
        rel_tokens = [f"<Relation: {self.relid2name[j]}>" for j in range(len(self.relid2name))]

        if mode != 'train' and self.ind == '_ind' and not self.full:
            with open(os.path.join(self.args.data_path, self.args.dataset, f'valid{self.ind}.txt'), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(self.args.data_path, self.args.dataset, f'test{self.ind}.txt'), 'r') as f:
                lines += f.readlines()
        else:
            with open(os.path.join(self.args.data_path, self.args.dataset, f'{mode}{self.ind}.txt'), 'r') as f:
                lines = f.readlines()
        hr2t = {}

        for line in lines:
            h, r, t = [int(x) for x in line.split()]
            if (h, r) not in hr2t:
                hr2t[(h, r)] = [t]
            else:
                hr2t[(h, r)].append(t)

        if mode != 'train':
            # extra_mode = 'test' if mode == 'valid' else 'valid'
            with open(os.path.join(self.args.data_path, self.args.dataset, f'train{self.ind}.txt'), 'r') as f:
                extra_lines = f.readlines()
            with open(os.path.join(self.args.data_path, self.args.dataset, f'valid{self.ind}.txt'), 'r') as f:
                extra_lines += f.readlines()
            if self.args.trans_or_ind == 'transductive' or self.ind == '_ind':
                with open(os.path.join(self.args.data_path, self.args.dataset, f'test{self.ind}.txt'), 'r') as f:
                    extra_lines += f.readlines()
            hr2t_extra = {}
            for line in extra_lines:
                h, r, t = [int(x) for x in line.split()]
                # triplets.append((h, t, r))
                if (h, r) not in hr2t_extra:
                    hr2t_extra[(h, r)] = [t]
                else:
                    hr2t_extra[(h, r)].append(t)
            alternative_t_extra = []

        h_kg_id, r_kg_id, t_kg_id, h_lql_id, r_lgl_id, t_lgl_id, input_text, input_text_inverse, alternative_t = [], [], [], [], [], [], [], [], []

        for line in tqdm(lines):
            h, r, t = [int(x) for x in line.split()]
            if 'inverse of' in self.relid2name[r]: # mode == 'train' and
                continue
            alternative_t.append(hr2t[(h, r)])
            if mode != 'train':
                alternative_t_extra.append(hr2t_extra.get((h, r), []))
            h_lql = f"<Entity: {self.entid2name[h]}>"
            r_lql = f"<Relation: {self.relid2name[r]}>"
            t_lql = f"<Entity: {self.entid2name[t]}>"
            text = self.get_text(h_lql, h, r_lql, r)
            h_kg_id.append(h)
            r_kg_id.append(r)
            t_kg_id.append(t)
            h_lql_id.append(self.org_vocab_size + h)
            r_lgl_id.append(self.org_vocab_size + self.num_ent + r)
            t_lgl_id.append(self.org_vocab_size + t)
            input_text.append(text)
            # if mode == 'train':
            inverse_r_lql = f"<Relation: {self.relid2name[r + int(self.num_rel // 2)]}>"
            inverse_text = self.get_text(t_lql, t, inverse_r_lql, r + int(self.num_rel // 2))
            input_text_inverse.append(inverse_text)
        if mode != 'train':
            data = pd.DataFrame({'h_kg_id': h_kg_id, 'r_kg_id': r_kg_id, 't_kg_id': t_kg_id,
                                 'h_lql_id': h_lql_id, 'r_lql_id': r_lgl_id, 't_lql_id': t_lgl_id,
                                 'input_text': input_text, 'input_text_inverse': input_text_inverse,
                                 'alternative_t': alternative_t, 'alternative_t_extra': alternative_t_extra})
        else:
            data = pd.DataFrame({'h_kg_id': h_kg_id, 'r_kg_id': r_kg_id, 't_kg_id': t_kg_id,
                                 'h_lql_id': h_lql_id, 'r_lql_id': r_lgl_id, 't_lql_id': t_lgl_id,
                                 'input_text': input_text, 'input_text_inverse': input_text_inverse,
                                 'alternative_t': alternative_t})

        ent2orgTokens = self.tokenizer(ent_tokens, max_length=self.cut_off, add_special_tokens=False,
                                       truncation=True, padding=True)
        rel2orgTokens = self.tokenizer(rel_tokens, max_length=self.cut_off, add_special_tokens=False,
                                       truncation=True, padding=True)
        ent2orgTokens = pd.DataFrame({'id_kg': list(range(len(ent_tokens))),
                                      'id_qlq': list(
                                          range(self.org_vocab_size, len(ent_tokens) + self.org_vocab_size)),
                                      'name': ent_tokens,
                                      'input_ids': ent2orgTokens.input_ids,
                                      'attention_mask': ent2orgTokens.attention_mask})
        rel2orgTokens = pd.DataFrame({'id_kg': list(range(len(rel_tokens))),
                                      'id_qlq': list(
                                          range(len(ent_tokens) + self.org_vocab_size, len(rel_tokens) + len(ent_tokens) + self.org_vocab_size)),
                                      'name': rel_tokens,
                                      'input_ids': rel2orgTokens.input_ids,
                                      'attention_mask': rel2orgTokens.attention_mask})
        if not os.path.exists(os.path.join(self.args.data_path, self.args.dataset, 'processed_data')):
            os.mkdir(os.path.join(self.args.data_path, self.args.dataset, 'processed_data'))
        with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_{mode}{self.ind}.pkl'), 'wb') as f:
            pickle.dump({'data': data, 'ent2orgTokens': ent2orgTokens, 'rel2orgTokens': rel2orgTokens}, f)
        return data
        # print()
        # train_edges = torch.tensor([[t[0], t[1]] for t in triplets], dtype=torch.long).t()
        # train_edge_types = torch.tensor([t[2] for t in triplets], dtype=torch.long)
        # graph = Data(edge_index=train_edges, edge_type=train_edge_types,
        #              num_nodes=len(self.ent2id), num_relations=len(self.rel2id))
        # graph = build_relation_graph(graph)
        # print()
    # def create_lql_from_kg(self):

if train_args().trans_or_ind == 'transductive':
    # KG(args=train_args())
    Create_LQL(args=train_args(), cut_off=10)
elif train_args().trans_or_ind == 'inductive':
    # KG(args=train_args())
    Create_LQL(args=train_args(), cut_off=10)
    # KG(args=train_args(), ind='_ind')
    Create_LQL(args=train_args(), cut_off=10, ind='_ind')
elif train_args().trans_or_ind == 'full_inductive':
    # KG(args=train_args())
    Create_LQL(args=train_args(), cut_off=10)
    # KG(args=train_args(), ind='_ind')
    Create_LQL(args=train_args(), cut_off=10, ind='_ind', full=True)