from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import argparse
from collections import defaultdict
import ast
import jinja2
from jinja2 import meta
from collections import OrderedDict

def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

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


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
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
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
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


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    # edge_index = [head_list, tail_list], edge_type = relation_list
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    # edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 5, 0, 0]]).to('cuda')
    # edge_type = torch.tensor([1, 0, 3, 2, 4, 0, 2]).to('cuda')
    # num_nodes, num_rels = 6, 5
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0) # remove duplicate (h, r)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0]) # Count the quantity of each head entity in Eh

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,  # (h, r)->(r, h)->[relation_list, head_list]
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]],  # edge weight 1/(head entity quantity)
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T,  # [head_list, relation_list]
        torch.ones(Eh.shape[0], device=device),  # edge weight 1
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0) # remove duplicate (t, r)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0]) # Count the quantity of each tail entity in Et
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T,   # (t, r)->(r, t)->[relation_list, tail_list]
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]],   # edge weight 1/(tail entity quantity)
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T,   # [tail_list, relation_list]
        torch.ones(Et.shape[0], device=device),  # edge weight 1
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    # hh_edges = torch.cat(
    #     [torch.nonzero(Ahh.to_dense()),
    #      torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0).to(Ahh.device)],
    #     dim=1)  # head to head
    # tt_edges = torch.cat(
    #     [torch.nonzero(Att.to_dense()),
    #      torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1).to(Ahh.device)],
    #     dim=1)  # tail to tail
    # ht_edges = torch.cat(
    #     [torch.nonzero(Aht.to_dense()), torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2).to(Ahh.device)],
    #     dim=1)  # head to tail
    # th_edges = torch.cat(
    #     [torch.nonzero(Ath.to_dense()), torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3).to(Ahh.device)],
    #     dim=1)  # tail to head

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0).to(Ahh.device)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1).to(Ahh.device)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2).to(Ahh.device)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3).to(Ahh.device)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph

def spreshape_back_32(sp):
    """
    Reshape a sparse tensor with signature (i,j,k) -> (i, j*k)
    """
    i, j, k = sp.indices()[0], sp.indices()[1], sp.indices()[2]
    val = sp.values()
    jk = j * sp.size(2) + k
    reshaped_indices = torch.stack((i, jk))
    return torch.sparse_coo_tensor(reshaped_indices, val, (sp.size(0), sp.size(1) * sp.size(2))).coalesce()

def spreshape_back_23(sp, k):
    """
    Reshape a sparse tensor with signature (i,j*k) -> (i, j, k)
    """
    input_k = k
    i, jk = sp.indices()[0], sp.indices()[1]
    val = sp.values()
    j = jk // k
    k = jk % k
    reshaped_indices = torch.stack((i, j, k))
    return torch.sparse_coo_tensor(reshaped_indices, val, (sp.size(0), sp.size(1) // input_k, input_k)).coalesce()

def spreshape_front_32(sp):
    """
    Reshape a sparse tensor with signature (i,j,k) -> (i*j,k)
    """
    i, j, k = sp.indices()[0], sp.indices()[1], sp.indices()[2]
    val = sp.values()
    ij = i * sp.size(1) + j
    reshaped_indices = torch.stack((ij, k))
    return torch.sparse_coo_tensor(reshaped_indices, val, (sp.size(0)*sp.size(1), sp.size(2))).coalesce()

def spreshape_front_23(sp, j):
    """
    Reshape a sparse tensor with signature (i*j,k) -> (i, j, k)
    """
    input_j = j
    ij, k = sp.indices()[0], sp.indices()[1]
    val = sp.values()
    i = ij // j
    j = ij % j
    reshaped_indices = torch.stack((i, j, k))
    return torch.sparse_coo_tensor(reshaped_indices, val, (sp.size(0) // input_j, input_j, sp.size(1))).coalesce()

def build_relation_hypergraph(graph, max_arity=3):
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device
    num_rels = torch.tensor(num_rels, device=device) if type(num_rels) is int else num_rels.clone().detach().to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,
        torch.ones(Eh.shape[0], device=device),

        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T,
        torch.ones(Eh.shape[0], device=device),
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T,
        torch.ones(Et.shape[0], device=device),

        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T,
        torch.ones(Et.shape[0], device=device),
        (num_nodes, num_rels)
    )

    forward_adj = torch.vstack([edge_index[0], edge_type, edge_index[1]]).to(device)  # E x R x E

    forward_adj = torch.sparse_coo_tensor(forward_adj,
                                          torch.ones(forward_adj.shape[1], device=device),
                                          (num_nodes, num_rels, num_nodes))
    forward_adj = forward_adj.coalesce()

    temp_tf = spreshape_front_32(
        spreshape_back_23(
            torch.sparse.mm(
                EtT,
                spreshape_back_32(forward_adj)
            ),
            num_nodes
        ))

    temp_hf = spreshape_front_32(
        spreshape_back_23(
            torch.sparse.mm(
                EhT,
                spreshape_back_32(forward_adj)
            ),
            num_nodes
        )
    )

    # Implement all 3-length path
    num_rels = torch.tensor(num_rels, device=device) if type(num_rels) is int else num_rels.clone().detach().to(device)

    Atfh = spreshape_front_23(torch.sparse.mm(temp_tf, Eh), num_rels)
    Atft = spreshape_front_23(torch.sparse.mm(temp_tf, Et), num_rels)
    Ahft = spreshape_front_23(torch.sparse.mm(temp_hf, Et), num_rels)
    Ahfh = spreshape_front_23(torch.sparse.mm(temp_hf, Eh), num_rels)

    tfh_edges = torch.cat(
        [Atfh.indices().T, torch.zeros(Atfh.indices().T.shape[0], 1, dtype=torch.long).fill_(3).to(device)], dim=1)
    tft_edges = torch.cat(
        [Atft.indices().T, torch.zeros(Atft.indices().T.shape[0], 1, dtype=torch.long).fill_(4).to(device)], dim=1)
    hft_edges = torch.cat(
        [Ahft.indices().T, torch.zeros(Ahft.indices().T.shape[0], 1, dtype=torch.long).fill_(5).to(device)], dim=1)
    hfh_edges = torch.cat(
        [Ahfh.indices().T, torch.zeros(Ahfh.indices().T.shape[0], 1, dtype=torch.long).fill_(6).to(device)], dim=1)

    edge_index_path_3 = torch.cat(
        [tfh_edges[:, [0, 1, 2]].T, tft_edges[:, [0, 1, 2]].T, hft_edges[:, [0, 1, 2]].T, hfh_edges[:, [0, 1, 2]].T],
        dim=1)
    edge_type_path_3 = torch.cat([tfh_edges[:, 3], tft_edges[:, 3], hft_edges[:, 3], hfh_edges[:, 3]], dim=0)
    num_relations_3 = 4

    # Implement all 2-length path
    # Note that since Aht and Ath will be identical under hypergraph setting as the pattern are isomorphic to each other

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()

    hh_edges = torch.cat(
        [Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0).to(device)],
        dim=1)  # head to head
    tt_edges = torch.cat(
        [Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1).to(device)],
        dim=1)  # tail to tail
    ht_edges = torch.cat(
        [Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2).to(device)],
        dim=1)  # head to tail

    edge_index_path_2 = torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T], dim=1)
    edge_type_path_2 = torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2]], dim=0)
    num_relations_2 = 3

    expanded_binary_edge_index = torch.zeros((max_arity, edge_index_path_2.size(1)), device=device,
                                             dtype=torch.long).fill_(
        -1)  # expand the arity to max_arity, -1 is padding node
    expanded_binary_edge_index[:2, :] = edge_index_path_2

    # Combine all edges
    rel_hypergraph = Data(
        edge_index=torch.cat([expanded_binary_edge_index, edge_index_path_3], dim=1),
        edge_type=torch.cat([edge_type_path_2, edge_type_path_3], dim=0),
        num_nodes=num_rels,
        num_relations=num_relations_2 + num_relations_3
    )

    graph.relation_hypergraph = rel_hypergraph
    return graph

def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            print(f'Trainable: {name}')
            trainable_params += num_params

    return trainable_params, all_param

def get_ht2r(triples):
    ht2r = defaultdict(lambda: set())
    for triple in triples:
        h, r, t = triple
        ht2r[(h, t)].add(r)
    return ht2r

def get_r2t(triples):
    r2t = defaultdict(lambda: set())
    for triple in triples:
        h, r, t = triple
        r2t[r].add(t)
    return r2t

def get_r2h(triples):
    r2h = defaultdict(lambda: set())
    for triple in triples:
        h, r, t = triple
        r2h[r].add(h)
    return r2h


def get_rel2triples(triples):
    rel2triples = defaultdict(lambda: set())
    for triple in triples:
        h, r, t = triple
        rel2triples[r].add(triple)
    for rel in rel2triples:
        rel2triples[rel] = list(rel2triples[rel])
    return rel2triples

def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars

def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ultra_config", help="yaml configuration file", type=str, default='../ultra_config/transductive/inference.yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the ultra_config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars

def load_ckpt(args, model):
    if args.init_ckpt != None:
        extra_state = OrderedDict()
        for k, v in torch.load(args.init_ckpt, map_location='cpu').items():
            extra_state[k.replace('module.', '')] = v
        new_state_dict = model.state_dict()
        for name, param in extra_state.items():
            new_state_dict[name].copy_(param)
        model.load_state_dict(new_state_dict)
    return model