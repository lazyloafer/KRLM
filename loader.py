import os
import pickle
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import easydict
import random
from torch_geometric.data import Data
from tasks import build_relation_graph, build_relation_hypergraph, all_negative
from functools import reduce
class DatasetBase():
    def __init__(self, args):
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
        self.lql_train, self.lql_valid, self.lql_test = self.load_lql()
        self.num_ent_train = len(self.lql_train['ent2orgTokens'])
        self.num_rel_train = len(self.lql_train['rel2orgTokens'])
        self.num_ent_test = len(self.lql_test['ent2orgTokens'])
        self.num_rel_test = len(self.lql_test['rel2orgTokens'])
        self.train_graph, self.valid_graph, self.test_graph = self.load_kg()

        self.tokenizer_train = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='./llm_source/Llama-2-7b-chat-hf',
            use_fast=False, add_eos_token=False)
        self.tokenizer_train.pad_token_id = 0
        self.tokenizer_train.padding_side = 'right'
        self.org_vocab_size_train = self.tokenizer_train.vocab_size

        self.ent2orgTokens_train = self.lql_train['ent2orgTokens']
        self.rel2orgTokens_train = self.lql_train['rel2orgTokens']

        all_entity_text_tokens_train = self.lql_train['ent2orgTokens'][['input_ids', 'attention_mask']].to_dict()
        entity2orgTokens_input_ids_train = []
        entity2orgTokens_attention_mask_train = []
        for key in range(len(all_entity_text_tokens_train['input_ids'])):
            entity2orgTokens_input_ids_train.append(all_entity_text_tokens_train['input_ids'][key])
            entity2orgTokens_attention_mask_train.append(all_entity_text_tokens_train['attention_mask'][key])
        self.entity2orgTokens_train = {'input_ids': torch.tensor(entity2orgTokens_input_ids_train).to(args.device),
                                 'attention_mask': torch.tensor(entity2orgTokens_attention_mask_train).unsqueeze(-1).to(args.device)}

        all_relation_text_tokens_train = self.lql_train['rel2orgTokens'][['input_ids', 'attention_mask']].to_dict()
        relation2orgTokens_input_ids_train = []
        relation2orgTokens_attention_mask_train = []
        for key in range(len(all_relation_text_tokens_train['input_ids'])):
            relation2orgTokens_input_ids_train.append(all_relation_text_tokens_train['input_ids'][key])
            relation2orgTokens_attention_mask_train.append(all_relation_text_tokens_train['attention_mask'][key])
        self.relation2orgTokens_train = {'input_ids': torch.tensor(relation2orgTokens_input_ids_train).to(args.device),
                                   'attention_mask': torch.tensor(relation2orgTokens_attention_mask_train).unsqueeze(-1).to(args.device)}

        self.entTokens_train = self.ent2orgTokens_train['name'].tolist()
        self.relTokens_train = self.rel2orgTokens_train['name'].tolist()

        self.tokenizer_train.add_tokens(['<Ent_Graph>', '<Rel_Graph>'] + self.entTokens_train + self.relTokens_train)
        ###############################################################################################
        self.tokenizer_test = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path='./llm_source/Llama-2-7b-chat-hf',
            use_fast=False, add_eos_token=False)
        self.tokenizer_test.pad_token_id = 0
        self.tokenizer_test.padding_side = 'right'
        self.org_vocab_size_test = self.tokenizer_test.vocab_size

        self.ent2orgTokens_test = self.lql_test['ent2orgTokens']
        self.rel2orgTokens_test = self.lql_test['rel2orgTokens']

        all_entity_text_tokens_test = self.lql_test['ent2orgTokens'][['input_ids', 'attention_mask']].to_dict()
        entity2orgTokens_input_ids_test = []
        entity2orgTokens_attention_mask_test = []
        for key in range(len(all_entity_text_tokens_test['input_ids'])):
            entity2orgTokens_input_ids_test.append(all_entity_text_tokens_test['input_ids'][key])
            entity2orgTokens_attention_mask_test.append(all_entity_text_tokens_test['attention_mask'][key])
        self.entity2orgTokens_test = {'input_ids': torch.tensor(entity2orgTokens_input_ids_test).to(args.device),
                                       'attention_mask': torch.tensor(entity2orgTokens_attention_mask_test).unsqueeze(
                                           -1).to(args.device)}

        all_relation_text_tokens_test = self.lql_test['rel2orgTokens'][['input_ids', 'attention_mask']].to_dict()
        relation2orgTokens_input_ids_test = []
        relation2orgTokens_attention_mask_test = []
        for key in range(len(all_relation_text_tokens_test['input_ids'])):
            relation2orgTokens_input_ids_test.append(all_relation_text_tokens_test['input_ids'][key])
            relation2orgTokens_attention_mask_test.append(all_relation_text_tokens_test['attention_mask'][key])
        self.relation2orgTokens_test = {'input_ids': torch.tensor(relation2orgTokens_input_ids_test).to(args.device),
                                         'attention_mask': torch.tensor(
                                             relation2orgTokens_attention_mask_test).unsqueeze(-1).to(args.device)}

        self.entTokens_test = self.ent2orgTokens_test['name'].tolist()
        self.relTokens_test = self.rel2orgTokens_test['name'].tolist()

        self.tokenizer_test.add_tokens(['<Ent_Graph>', '<Rel_Graph>'] + self.entTokens_test + self.relTokens_test)

    def load_lql(self):
        with open(
                os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_train.pkl'),
                'rb') as f:
            lql_train = pickle.load(f)
        with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_valid.pkl'),
                  'rb') as f:
            lql_valid = pickle.load(f)
        if self.args.dataset in ['WN18RR', 'CoDEx_M', 'FB15k237']:
            with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_test.pkl'),
                      'rb') as f:
                lql_test = pickle.load(f)
        else:
            with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_test_ind.pkl'),
                      'rb') as f:
                lql_test = pickle.load(f)
        return lql_train, lql_valid, lql_test

    def load_kg(self):
        # heads = torch.tensor(self.lql_train['data'][['h_kg_id']].values, dtype=torch.long).t()
        # rels = torch.tensor(self.lql_train['data'][['r_kg_id']].values, dtype=torch.long).t()
        # tails = torch.tensor(self.lql_train['data'][['t_kg_id']].values, dtype=torch.long).t()

        heads = torch.concat([torch.tensor(self.lql_train['data'][['h_kg_id']].values, dtype=torch.long).t(),
                              torch.tensor(self.lql_train['data'][['t_kg_id']].values, dtype=torch.long).t()], dim=-1)
        rels = torch.concat([torch.tensor(self.lql_train['data'][['r_kg_id']].values, dtype=torch.long).t(),
                             torch.tensor(self.lql_train['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                          dtype=torch.long).t()], dim=-1)
        tails = torch.concat([torch.tensor(self.lql_train['data'][['t_kg_id']].values, dtype=torch.long).t(),
                              torch.tensor(self.lql_train['data'][['h_kg_id']].values, dtype=torch.long).t()], dim=-1)
        train_triplets = torch.concat([heads, rels, tails], dim=0)
        # train_triplets = torch.tensor(self.lql_train['data'][['h_kg_id', 'r_kg_id', 't_kg_id']].values,
        #                               dtype=torch.long).t()
        train_edges = train_triplets[[0, 2]]
        train_edge_types = train_triplets[1]
        train_graph = Data(edge_index=train_edges, edge_type=train_edge_types,
                           num_nodes=len(self.lql_train['ent2orgTokens']),
                           num_relations=len(self.lql_train['rel2orgTokens']),
                           # inverse_rel_plus_one=True
                           )
        train_graph = build_relation_graph(train_graph)
        train_graph = build_relation_hypergraph(train_graph)
        valid_graph = train_graph

        if self.args.dataset in ['WN18RR', 'CoDEx_M', 'FB15k237']:
            test_graph = train_graph
        else:
            with open(
                    os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_train_ind.pkl'),
                    'rb') as f:
                lql_train_ind = pickle.load(f)
            heads = torch.concat([torch.tensor(lql_train_ind['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_train_ind['data'][['t_kg_id']].values, dtype=torch.long).t()],
                                 dim=-1)
            rels = torch.concat([torch.tensor(lql_train_ind['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_train_ind['data'][['r_kg_id']].values + (self.num_rel_test // 2),
                                              dtype=torch.long).t()],
                                dim=-1)
            tails = torch.concat([torch.tensor(lql_train_ind['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_train_ind['data'][['h_kg_id']].values, dtype=torch.long).t()],
                                 dim=-1)
            test_triplets = torch.concat([heads, rels, tails], dim=0)
            test_edges = test_triplets[[0, 2]]
            test_edge_types = test_triplets[1]
            test_graph = Data(edge_index=test_edges, edge_type=test_edge_types,
                              num_nodes=len(lql_train_ind['ent2orgTokens']),
                              num_relations=len(lql_train_ind['rel2orgTokens']), inverse_rel_plus_one=True)
            test_graph = build_relation_graph(test_graph)
            test_graph = build_relation_hypergraph(test_graph)

        return train_graph, valid_graph, test_graph

class LQLDatasetTrain(DatasetBase, Dataset):
    def __init__(self, args, mode):
        super().__init__(args)
        self.args = args
        self.mode = mode
        self.data = self.lql_train['data']#[:100]
        self.negative_sample_num = args.negative_sample_num
        self.all_entity_ids = list(range(len(self.ent2orgTokens_train)))

    def __getitem__(self, item):
        h_kg_id = int(self.data['h_kg_id'][item])
        r_kg_id = int(self.data['r_kg_id'][item])
        t_kg_id = int(self.data['t_kg_id'][item])
        alternative_t = self.data['alternative_t'][item]
        # negative_samples = random.sample(set(self.all_entity_ids).difference(set(alternative_t)),
        #                                  self.negative_sample_num)
        # t_kg_id = [t_kg_id] + negative_samples

        return {'h_kg_id': h_kg_id,
                'r_kg_id': r_kg_id,
                't_kg_id': t_kg_id,
                'h_lql_id': self.data['h_lql_id'][item],
                'r_lql_id': self.data['r_lql_id'][item],
                't_lql_id': self.data['t_lql_id'][item],
                'input_text': self.data['input_text'][item],
                'input_text_inverse': self.data['input_text_inverse'][item],
                'alternative_t': alternative_t,
                'head2text_inputs': self.ent2orgTokens_train.loc[h_kg_id][['input_ids', 'attention_mask']].to_dict(),
                'rel2text_inputs': self.rel2orgTokens_train.loc[r_kg_id][['input_ids', 'attention_mask']].to_dict(),
                'tail2text_inputs': self.ent2orgTokens_train.loc[t_kg_id][['input_ids', 'attention_mask']].to_dict(),
                'rel2text_inputs_inverse': self.rel2orgTokens_train.loc[r_kg_id + int(self.num_rel_train // 2)][['input_ids', 'attention_mask']].to_dict()
                # 'target2text_inputs': target2text_inputs
        }

    def __len__(self):
        return len(self.data)

class LQLDatasetValid(DatasetBase, Dataset):
    def __init__(self, args, mode, test_fast=False):
        super().__init__(args)
        self.args = args
        self.mode = mode
        self.filtered_data = self.load_filtered_data()
        # if self.args.dataset in ['WN18RR', 'CoDEx_M', 'FB15k237']:
        #     self.data = self.lql_test['data'][:3000]
        # else:
        #     self.data = self.lql_test['data']
        if test_fast:
            self.data = self.lql_test['data']#[:100]
        else:
            self.data = self.lql_test['data']#[:100]
        self.all_entity_ids = list(range(len(self.ent2orgTokens_test)))


    def load_filtered_data(self):
        if self.args.dataset in ['WN18RR', 'CoDEx_M', 'FB15k237']:
            with open(
                    os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_train.pkl'),
                    'rb') as f:
                lql_train = pickle.load(f)
            with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_valid.pkl'),
                      'rb') as f:
                lql_valid = pickle.load(f)
            with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_test.pkl'),
                      'rb') as f:
                lql_test = pickle.load(f)
            heads = torch.concat([torch.tensor(lql_train['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_train['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['t_kg_id']].values, dtype=torch.long).t()],
                                 dim=-1)
            rels = torch.concat([torch.tensor(lql_train['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_train['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                              dtype=torch.long).t(),
                                 torch.tensor(lql_valid['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_valid['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                              dtype=torch.long).t(),
                                 torch.tensor(lql_test['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_test['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                              dtype=torch.long).t()],
                                dim=-1)
            tails = torch.concat([torch.tensor(lql_train['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_train['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['h_kg_id']].values, dtype=torch.long).t()],
                                 dim=-1)
            filtered_triplets = torch.concat([heads, rels, tails], dim=0)
            filtered_edges = filtered_triplets[[0, 2]]
            filtered_edge_types = filtered_triplets[1]
            filtered_data = Data(edge_index=filtered_edges, edge_type=filtered_edge_types, num_nodes=self.num_ent_train)
        else:
            with open(
                    os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_train_ind.pkl'),
                    'rb') as f:
                lql_train = pickle.load(f)
            with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_valid_ind.pkl'),
                      'rb') as f:
                lql_valid = pickle.load(f)
            with open(os.path.join(self.args.data_path, self.args.dataset, 'processed_data', f'lql_from_kg_test_ind.pkl'),
                      'rb') as f:
                lql_test = pickle.load(f)
            heads = torch.concat([torch.tensor(lql_train['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_train['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['t_kg_id']].values, dtype=torch.long).t()],
                                 dim=-1)
            rels = torch.concat([torch.tensor(lql_train['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_train['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                              dtype=torch.long).t(),
                                 torch.tensor(lql_valid['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_valid['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                              dtype=torch.long).t(),
                                 torch.tensor(lql_test['data'][['r_kg_id']].values, dtype=torch.long).t(),
                                 torch.tensor(lql_test['data'][['r_kg_id']].values + (self.num_rel_train // 2),
                                              dtype=torch.long).t()],
                                dim=-1)
            tails = torch.concat([torch.tensor(lql_train['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_train['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_valid['data'][['h_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['t_kg_id']].values, dtype=torch.long).t(),
                                  torch.tensor(lql_test['data'][['h_kg_id']].values, dtype=torch.long).t()],
                                 dim=-1)
            filtered_triplets = torch.concat([heads, rels, tails], dim=0)
            filtered_edges = filtered_triplets[[0, 2]]
            filtered_edge_types = filtered_triplets[1]
            filtered_data = Data(edge_index=filtered_edges, edge_type=filtered_edge_types, num_nodes=self.num_ent_test)
        return filtered_data
    def __getitem__(self, item):
        h_kg_id = int(self.data['h_kg_id'][item])
        r_kg_id = int(self.data['r_kg_id'][item])
        t_kg_id = int(self.data['t_kg_id'][item])
        alternative_t = self.data['alternative_t'][item]
        # alternative_t_extra = self.data['alternative_t_extra'][item]

        alternative_t_extra = torch.ones(len(self.all_entity_ids), dtype=torch.bool)
        alternative_t_extra[self.data['alternative_t_extra'][item]] = False
        # alternative_t_extra[t_kg_id] = True
        alternative_t_extra = alternative_t_extra.unsqueeze(0)
        # t_kg_id = self.data['t_kg_id'][item]
        # h_lql_id = self.data['h_lql_id'][item]
        # r_lgl_id = self.data['r_lgl_id'][item]
        # t_lgl_id = self.data['t_lgl_id'][item]
        # input_text = self.data['input_text'][item]
        # alternative_t = self.data['alternative_t'][item]
        #
        # ent2text_inputs = self.ent2orgTokens.loc[h_kg_id][['input_ids', 'attention_mask']].to_dict()
        # rel2text_inputs = self.rel2orgTokens.loc[r_kg_id][['input_ids', 'attention_mask']].to_dict()
        # target2text_inputs = self.ent2orgTokens.loc[t_kg_id][['input_ids', 'attention_mask']].to_dict()
        return {'h_kg_id': h_kg_id,
                'r_kg_id': r_kg_id,
                't_kg_id': t_kg_id,
                'h_lql_id': self.data['h_lql_id'][item],
                'r_lql_id': self.data['r_lql_id'][item],
                't_lql_id': self.data['t_lql_id'][item],
                'input_text': self.data['input_text'][item],
                'input_text_inverse': self.data['input_text_inverse'][item],
                'alternative_t': alternative_t,
                'alternative_t_extra': alternative_t_extra,
                'ent2text_inputs': self.ent2orgTokens_test.loc[h_kg_id][['input_ids', 'attention_mask']].to_dict(),
                'rel2text_inputs': self.rel2orgTokens_test.loc[r_kg_id][['input_ids', 'attention_mask']].to_dict(),
                'target2text_inputs': self.ent2orgTokens_test.loc[self.all_entity_ids][['input_ids', 'attention_mask']].to_dict()}

    def __len__(self):
        return len(self.data)

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: [max_node_num, max_relation_num]
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)  # [max_node_num, max_node_num * max_relation_num]
    scale = scale[-1] // scale  # [max_relation_num, 1]

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type]).to(batch.device)
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type]).to(batch.device)
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask

def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)

def collate_train(samples, args, graph, tokenizer):
    batch = len(samples)
    (h_kg_id, \
    r_kg_id, \
    r_kg_id_inverse, \
    t_kg_id, \
    h_lql_id, \
    r_lql_id, \
    t_lql_id, \
    input_text, \
    input_text_inverse, \
    alternative_t, \
    head2text_input_ids, \
    head2text_attention_mask, \
    rel2text_input_ids, \
    rel2text_attention_mask, \
    rel2text_input_ids_inverse, \
    rel2text_attention_mask_inverse, \
    target2text_input_ids, \
    target2text_attention_mask) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for i in range(batch):
        h_kg_id.append(samples[i]['h_kg_id'])
        r_kg_id.append(samples[i]['r_kg_id'])
        r_kg_id_inverse.append(samples[i]['r_kg_id'] + int(graph.num_relations // 2))
        t_kg_id.append(samples[i]['t_kg_id'])
        # t_kg_id.append(samples[i]['t_kg_id'])
        # h_lql_id.append(samples[i]['h_lql_id'])
        # r_lql_id.append(samples[i]['r_lql_id'])
        # t_lql_id.append(samples[i]['t_lql_id'])
        input_text.append(samples[i]['input_text'])
        input_text_inverse.append(samples[i]['input_text_inverse'])
        # alternative_t.append(samples[i]['alternative_t'])
        # head2text_input_ids.append(samples[i]['head2text_inputs']['input_ids'])
        # head2text_attention_mask.append(samples[i]['head2text_inputs']['attention_mask'])
        # rel2text_input_ids.append(samples[i]['rel2text_inputs']['input_ids'])
        # rel2text_attention_mask.append(samples[i]['rel2text_inputs']['attention_mask'])
        # rel2text_input_ids_inverse.append(samples[i]['rel2text_inputs_inverse']['input_ids'])
        # rel2text_attention_mask_inverse.append(samples[i]['rel2text_inputs_inverse']['attention_mask'])
        # target2text_input_ids.append(samples[i]['tail2text_inputs']['input_ids'])
        # target2text_attention_mask.append(samples[i]['tail2text_inputs']['attention_mask'])

        # target2text_input_ids.append(samples[i]['target2text_inputs']['input_ids'])
        # target2text_attention_mask.append(samples[i]['target2text_inputs']['attention_mask'])

        # for k in range(len(samples[i]['t_kg_id'])):
        #     key = samples[i]['t_kg_id'][k]
        #     target2text_input_ids.append(samples[i]['target2text_inputs']['input_ids'][key])
        #     target2text_attention_mask.append(samples[i]['target2text_inputs']['attention_mask'][key])

    # triplets = torch.tensor([h_kg_id + h_kg_id, t_kg_id + t_kg_id, r_kg_id + r_kg_id]).T
    triplets = torch.tensor([h_kg_id, t_kg_id, r_kg_id]).T
    triplets = negative_sampling(graph, triplets, args.negative_sample_num)  # [:, 0]

    input_text = tokenizer(input_text[:batch//2] + input_text_inverse[batch//2:], return_tensors='pt', padding=True) # , truncation=True
    input_length = torch.sum(input_text.attention_mask, dim=1)

    batch = {'triplets': triplets,
             'h_kg_id': torch.tensor(h_kg_id[:batch//2] + t_kg_id[batch//2:]),
            'r_kg_id': torch.tensor(r_kg_id[:batch//2] + r_kg_id_inverse[batch//2:]),
            't_kg_id': torch.tensor(t_kg_id[:batch//2] + h_kg_id[batch//2:]),
            # 'h_lql_id': torch.tensor(h_lql_id),
            # 'r_lql_id': torch.tensor(r_lql_id),
            # 't_lql_id': torch.tensor(t_lql_id),
            'input_text': input_text,
            'input_length': input_length,
            # 'alternative_t': alternative_t,
            # 'ent2text_input_ids': torch.tensor(head2text_input_ids + target2text_input_ids),
            # 'ent2text_attention_mask': torch.tensor(head2text_attention_mask + target2text_attention_mask),
            # 'rel2text_input_ids': torch.tensor(rel2text_input_ids + rel2text_input_ids_inverse),
            # 'rel2text_attention_mask': torch.tensor(rel2text_attention_mask + rel2text_attention_mask_inverse)
    }
    return easydict.EasyDict(batch)

def collate_test(samples, args, graph, tokenizer):
    batch = len(samples)
    num_ent = graph.num_nodes
    h_kg_id, \
    r_kg_id, \
    r_kg_id_inverse, \
    t_kg_id, \
    h_lql_id, \
    r_lql_id, \
    t_lql_id, \
    input_text, \
    input_text_inverse, \
    alternative_t, \
    alternative_t_extra, \
    ent2text_input_ids, \
    ent2text_attention_mask, \
    rel2text_input_ids, \
    rel2text_attention_mask, \
    target2text_input_ids, \
    target2text_attention_mask = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for i in range(batch):

        h_kg_id.append(samples[i]['h_kg_id'])
        r_kg_id.append(samples[i]['r_kg_id'])
        r_kg_id_inverse.append(samples[i]['r_kg_id'] + int(graph.num_relations // 2))
        t_kg_id.append(samples[i]['t_kg_id'])
        # h_lql_id.append(samples[i]['h_lql_id'])
        # r_lql_id.append(samples[i]['r_lql_id'])
        # t_lql_id.append(samples[i]['t_lql_id'])
        input_text.append(samples[i]['input_text'])
        input_text_inverse.append(samples[i]['input_text_inverse'])
        # alternative_t.append(samples[i]['alternative_t'])
        # alternative_t_extra.append(samples[i]['alternative_t_extra'])
        # alternative_t_extra.append(samples[i]['alternative_t_extra'])
        # ent2text_input_ids.append(samples[i]['ent2text_inputs']['input_ids'])
        # ent2text_attention_mask.append(samples[i]['ent2text_inputs']['attention_mask'])
        # rel2text_input_ids.append(samples[i]['rel2text_inputs']['input_ids'])
        # rel2text_attention_mask.append(samples[i]['rel2text_inputs']['attention_mask'])

        # for key in range(len(samples[i]['target2text_inputs']['input_ids'])):
        #     target2text_input_ids.append(samples[i]['target2text_inputs']['input_ids'][key])
        #     target2text_attention_mask.append(samples[i]['target2text_inputs']['attention_mask'][key])

    triplets = torch.tensor([h_kg_id, t_kg_id, r_kg_id]).T
    t_batch, h_batch = all_negative(graph, triplets)
    input_text = tokenizer(input_text, return_tensors='pt', padding=True) # , truncation=True
    input_length = torch.sum(input_text.attention_mask, dim=1)

    input_text_inverse = tokenizer(input_text_inverse, return_tensors='pt', padding=True)  # , truncation=True
    input_length_inverse = torch.sum(input_text_inverse.attention_mask, dim=1)

    t_batch = {'triplets': t_batch,
             'h_kg_id': torch.tensor(h_kg_id),
            'r_kg_id': torch.tensor(r_kg_id),
            't_kg_id': torch.tensor(t_kg_id),
            # 'h_lql_id': torch.tensor(h_lql_id),
            # 'r_lql_id': torch.tensor(r_lql_id),
            # 't_lql_id': torch.tensor(t_lql_id),
            'input_text': input_text,
            'input_length': input_length,
            # 'alternative_t': alternative_t,
            # 'alternative_t_extra': torch.concat(alternative_t_extra, dim=0),
            # 'ent2text_input_ids': torch.tensor(ent2text_input_ids),
            # 'ent2text_attention_mask': torch.tensor(ent2text_attention_mask),
            # 'rel2text_input_ids': torch.tensor(rel2text_input_ids),
            # 'rel2text_attention_mask': torch.tensor(rel2text_attention_mask),
            # 'target2text_input_ids': torch.tensor(target2text_input_ids),
            # 'target2text_attention_mask': torch.tensor(target2text_attention_mask)
    }

    h_batch = {'triplets': h_batch,
               'h_kg_id': torch.tensor(t_kg_id),
               'r_kg_id': torch.tensor(r_kg_id_inverse),
               't_kg_id': torch.tensor(h_kg_id),
               # 'h_lql_id': torch.tensor(h_lql_id),
               # 'r_lql_id': torch.tensor(r_lql_id),
               # 't_lql_id': torch.tensor(t_lql_id),
               'input_text': input_text_inverse,
               'input_length': input_length_inverse,
               # 'alternative_t': alternative_t,
               # 'alternative_t_extra': torch.concat(alternative_t_extra, dim=0),
               # 'ent2text_input_ids': torch.tensor(ent2text_input_ids),
               # 'ent2text_attention_mask': torch.tensor(ent2text_attention_mask),
               # 'rel2text_input_ids': torch.tensor(rel2text_input_ids),
               # 'rel2text_attention_mask': torch.tensor(rel2text_attention_mask),
               # 'target2text_input_ids': torch.tensor(target2text_input_ids),
               # 'target2text_attention_mask': torch.tensor(target2text_attention_mask)
               }
    return triplets, easydict.EasyDict(t_batch), easydict.EasyDict(h_batch)

def multigraph_collator(batch, data_train_list, args, tokenizer_train_list):
    probs = torch.tensor([data_train.train_graph.edge_index.shape[1] for data_train in data_train_list]).float()
    probs /= probs.sum()
    graph_id = torch.multinomial(probs, 1, replacement=False).item()
    tokenizer = tokenizer_train_list[graph_id]
    data_train = data_train_list[graph_id]
    entity2orgTokens_train = data_train.entity2orgTokens_train
    relation2orgTokens_train = data_train.relation2orgTokens_train
    data_train_dataframe = data_train.lql_train['data']
    bs = len(batch)
    data_mask = torch.randperm(len(data_train_dataframe))[:bs]
    batch_data = data_train_dataframe.loc[data_mask, :]
    triplets = torch.tensor(batch_data.loc[:, ['h_kg_id', 't_kg_id', 'r_kg_id']].values.tolist())
    h_kg_id = torch.cat([triplets[:int(bs // 2), 0], triplets[int(bs // 2):, 1]])
    t_kg_id = torch.cat([triplets[:int(bs // 2), 1], triplets[int(bs // 2):, 0]])
    r_kg_id = torch.cat([triplets[:int(bs // 2), 2], triplets[int(bs // 2):, 2] + int(data_train.train_graph.num_relations // 2)])
    triplets = negative_sampling(data_train.train_graph, triplets, args.negative_sample_num)

    input_text = [s[0] for s in batch_data.loc[:, ['input_text']].values.tolist()]
    input_text_inverse = [s[0] for s in batch_data.loc[:, ['input_text_inverse']].values.tolist()]

    input_text = tokenizer(input_text[:bs // 2] + input_text_inverse[bs // 2:], return_tensors='pt', padding=True)  # , truncation=True
    input_length = torch.sum(input_text.attention_mask, dim=1)

    batch = {'triplets': triplets,
             'h_kg_id': h_kg_id,
             'r_kg_id': r_kg_id,
             't_kg_id': t_kg_id,
             # 'h_lql_id': torch.tensor(h_lql_id),
             # 'r_lql_id': torch.tensor(r_lql_id),
             # 't_lql_id': torch.tensor(t_lql_id),
             'input_text': input_text,
             'input_length': input_length,
             # 'alternative_t': alternative_t,
             # 'ent2text_input_ids': torch.tensor(head2text_input_ids + target2text_input_ids),
             # 'ent2text_attention_mask': torch.tensor(head2text_attention_mask + target2text_attention_mask),
             # 'rel2text_input_ids': torch.tensor(rel2text_input_ids + rel2text_input_ids_inverse),
             # 'rel2text_attention_mask': torch.tensor(rel2text_attention_mask + rel2text_attention_mask_inverse)
             }
    return easydict.EasyDict(batch), data_train.train_graph, tokenizer, entity2orgTokens_train, relation2orgTokens_train


