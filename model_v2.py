import copy
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Tuple, Union, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
# from motif.models import MOTIF
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from KGFM.models import EntityNBFNet
class TextAggregator(nn.Module):
    def __init__(self, args, llm_hidden_dim, aggregate_func, is_target=False):
        super(TextAggregator, self).__init__()
        self.args = args
        self.llm_hidden_dim = llm_hidden_dim
        self.aggregate_func = aggregate_func
        self.device = self.args.device
        self.is_target = is_target

        self.down_scaling = nn.Linear(
            self.llm_hidden_dim, self.args.dim, bias=False, dtype=torch.float)
        if self.aggregate_func == 'pna':
            self.re_scaling = nn.Linear(self.args.dim * 12, self.args.dim)   # 12, 15
        # self.up_scaling_relation = nn.Linear(
        #     self.args.dim * 2, self.llm_hidden_dim, bias=False, dtype=torch.float)

    def norm(self, x):
        return F.normalize(x, p=2, dim=1)

    def forward(self, text_embedding, token_ids, token_mask, additional_embedding=None):

        text_embedding = self.down_scaling(text_embedding)
        token_lengths = token_mask.half().sum(axis=1).to(self.device)  # B X 1
        degree = token_lengths
        token_embs = text_embedding[token_ids]  # B x L x Hidden

        mean = (token_embs * token_mask).sum(axis=1) / token_lengths
        if self.aggregate_func == 'mean':
            text_output = mean
        elif self.aggregate_func == 'sum':
            text_output = (token_embs * token_mask).sum(axis=1)
        elif self.aggregate_func == 'pna':
            sq_mean = (token_embs ** 2 * token_mask).sum(axis=1) / token_lengths
            max, _ = (token_embs * token_mask).max(axis=1)
            min, _ = (token_embs * token_mask).min(axis=1)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            features = torch.cat([mean, max, min, std], dim=-1)
            # if additional_embedding != None:
            #     if self.is_target:
            #         features = features.unsqueeze(0).repeat(additional_embedding.size(0), 1, 1)
            #     features = torch.concatenate([features, additional_embedding], dim=-1)
            # if self.is_target:
            #     degree = degree.unsqueeze(0).repeat(additional_embedding.size(0), 1, 1)
            scale = degree.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)

            text_output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError('aggregate_func = [mean, sum, pna]')

        # # if self.is_target:
        # text_output = text_output.unsqueeze(0).repeat(additional_embedding.size(0), 1, 1)
        if self.aggregate_func == 'pna':
            # text_output = self.re_scaling(torch.cat([text_output, additional_embedding], dim=-1))
            text_output = self.re_scaling(text_output)

        text_output = self.norm(text_output)

        return text_output

class AttE(nn.Module):
    def __init__(self, args):
        super(AttE, self).__init__()
        self.args = args
        # self.c_layer = nn.Linear(args.dim, 1, bias=False)
        self.rel_diag_layer = nn.Linear(args.dim, args.dim * 2, bias=False)
        self.context_vec_layer = nn.Linear(args.dim, args.dim, bias=False)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(args.dim)]).cuda()
    def givens_rotations(self, r, x):
        r = r.unsqueeze(1).repeat(1, x.size(1), 1)
        givens = r.view((r.size(0), r.size(1), -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((x.size(0), x.size(1), -1, 2))
        x_rot = givens[:, :, :, 0:1] * x + givens[:, :, :, 1:] * torch.cat((-x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1)
        return x_rot.view((x.size(0), x.size(1), -1))

    def given_reflection(self, r, x):
        r = r.unsqueeze(1).repeat(1, x.size(1), 1)
        givens = r.view((r.size(0), r.size(1), -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((x.size(0), x.size(1), -1, 2))
        x_ref = givens[:, :, :, 0:1] * torch.cat((x[:, :, :, 0:1], -x[:, :, :, 1:]), dim=-1) + givens[:, :, :, 1:] * torch.cat((x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1)
        return x_ref.view((x.size(0), x.size(1), -1))

    def project(self, x, c):
        BALL_EPS = {torch.float32: 4e-3,  torch.float64: 1e-5}
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-15)
        eps = BALL_EPS[x.dtype]
        maxnorm = (1 - eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def expmap(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-15)
        gamma_1 = (sqrt_c * u_norm).clamp(-15, 15).tanh() * u / (sqrt_c * u_norm)
        return self.project(gamma_1, c)

    def mobius_add(self, x, y, c):
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        demon = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / demon.clamp_min(1e-15)

    def score(self, x, v):
        # sqrt_c = c ** 5
        # vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        # xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
        #
        # gamma = (sqrt_c * vnorm).clamp(-15, 15).tanh() / sqrt_c
        # x2 = torch.sum(x * x, dim=-1, keepdim=True)
        # c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
        # c2 = 1 - c * x2
        # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
        # denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
        # pairwise_norm = num / denom.clamp_min(1e-15)
        # dist = Artanh.apply(sqrt_c * pairwise_norm)
        # return - (2 * dist / sqrt_c) ** 2
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        v2 = torch.sum(v * v, dim=-1, keepdim=True)
        xv = torch.sum(x * v, dim=-1, keepdim=True)
        score = - torch.sqrt(x2 + v2 - 2 * xv).squeeze(-1)
        mask = - torch.ones_like(score)
        mask[:, 0] = 1
        score = torch.mean(torch.sum(- F.logsigmoid(score * mask), dim=-1))
        return score

    def forward(self, nbf_ent_output, nbf_rel_output, h_kg_id, r_kg_id, t_index):  # inputs = BxLxdim, query_rel_embeddings = Bxdim
        self.bs = nbf_ent_output.size(0)
        # query_head = nbf_ent_output[torch.arange(self.bs), h_kg_id]
        query_rel = nbf_rel_output[torch.arange(self.bs), r_kg_id]
        # hyper_c = F.softplus(self.c_layer(query_rel)) # Bx1
        rot_mat, ref_mat = torch.chunk(self.rel_diag_layer(query_rel), 2, dim=1) # Bxdim
        rot_q = self.givens_rotations(rot_mat, nbf_ent_output).unsqueeze(1)
        ref_q = self.given_reflection(ref_mat, nbf_ent_output).unsqueeze(1)
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec_layer(query_rel.unsqueeze(1).repeat(1, nbf_ent_output.size(1), 1))
        att_weights = torch.sum(context_vec.unsqueeze(1) * cands * self.scale, dim=-1, keepdim=True)
        att_ent_output = torch.sum(self.act(att_weights) * cands, dim=1)
        # hyper_ent_output = self.expmap(att_ent_output, hyper_c.unsqueeze(1).repeat(1, nbf_ent_output.size(1), 1))

        # hyper_rel_output = self.expmap(nbf_rel_output, F.softplus(self.c_layer(nbf_rel_output)))
        if self.training:
            att_res = att_ent_output[torch.arange(self.bs), h_kg_id] + query_rel
            # hyper_res = self.project(
            #     self.mobius_add(
            #         att_ent_output[torch.arange(self.bs), h_kg_id],
            #         hyper_rel_output[torch.arange(self.bs), r_kg_id],
            #         hyper_c
            #     ),
            #     hyper_c
            # ).unsqueeze(1).repeat(1, t_index.size(1), 1)
            targets = nbf_ent_output[
                torch.arange(self.bs).repeat_interleave(t_index.size(1), dim=0),
                torch.flatten(t_index)
            ].view(self.bs, t_index.size(1), -1)
            # att_score = torch.sum(att_res.unsqueeze(1).repeat(1, t_index.size(1), 1) * targets, dim=-1, keepdim=True).squeeze(-1)
            att_score = self.score(att_res.unsqueeze(1).repeat(1, t_index.size(1), 1), targets)
        else:
            att_score = 0
        return att_ent_output, nbf_rel_output, att_score
class LQLModel(nn.Module):

    def __init__(self, args, mofit_cfg, llm=None, kgmodel=None, dropout_ratio=0.25, more_dropout=0.0):
        super(LQLModel, self).__init__()
        self.args = args
        self.llm = llm
        self.kgmodel = kgmodel
        # self.AttH = AttE(args)
        self.device = args.device
        self.dropout_ratio = dropout_ratio
        self.more_dropout = more_dropout
        self.remove_one_hop = False
        self.aggregate_func_nbf = self.args.aggregate_func_nbf
        self.aggregate_func_text = self.args.aggregate_func_text
        self.llm_hidden_dim = self.llm.get_input_embeddings().weight.shape[1]
        # self.tail_context_retriever = copy.deepcopy(self.kgmodel.entity_model)
        # for name, param in self.tail_context_retriever.named_parameters():  # 26,291,138
        #     param.requires_grad = True
        self.tail_context_retriever = EntityNBFNet(is_target=True, **mofit_cfg.model.entity_model)
        self.text_aggregator_head = TextAggregator(args=args,
                                                   llm_hidden_dim=self.llm_hidden_dim,
                                                   aggregate_func=self.aggregate_func_text)
        # self.head_weight = nn.Linear(self.args.dim * 2, 2, bias=True)
        self.text_aggregator_tail = TextAggregator(args=args,
                                                   llm_hidden_dim=self.llm_hidden_dim,
                                                   aggregate_func=self.aggregate_func_text,
                                                   is_target=True)
        # self.tail_weight = nn.Linear(self.args.dim * 2, 2, bias=True)
        # self.down_scaling = nn.Linear(
        #     self.llm_hidden_dim, self.args.dim, bias=False, dtype=torch.float)
        # if self.aggregate_func == 'pna':
        #     self.re_scaling = nn.Linear(self.args.dim * 12, self.args.dim)
        self.up_scaling_ent_rel_token = nn.Linear(self.args.dim, self.llm_hidden_dim, bias=True)
        self.up_scaling_ent_rel_graph = nn.Linear(self.args.dim, self.llm_hidden_dim, bias=True)
        # self.down_scaling_target = nn.Linear(
        #     self.args.dim, 1, bias=False, dtype=torch.float)
        # mlp = [nn.Linear(self.args.dim, self.args.dim), nn.ReLU(), nn.Linear(self.args.dim, 1)]
        # self.down_scaling_target = nn.Sequential(*mlp)
        self.down_scaling_hidden = nn.Linear(
            self.llm_hidden_dim, self.args.dim, bias=True)
        # self.init_score_mlp = nn.Linear(
        #     3 * self.args.dim, self.args.dim, bias=False, dtype=torch.float)
        self.mlp = nn.Sequential()
        mlp = [nn.Linear(self.args.dim * 3, self.args.dim),
               nn.ReLU(),
               nn.Linear(self.args.dim, 1)]
        self.mlp = nn.Sequential(*mlp)

    def get_ent_rel_embedding(self,
                              text_embedding,
                              ent_rel_ids,
                              token_ids,
                              token_mask,
                              text_aggregator,
                              nbf_ent_rel_embedding=None,
                              hyper_ent_rel_embedding=None):
        # ent_rel_embedding = text_aggregator(text_embedding, token_ids[ent_rel_ids], token_mask[ent_rel_ids], additional_embedding=query_nbf_ent_rel_embedding)
        text_ent_rel_embedding = text_aggregator(text_embedding, token_ids, token_mask, additional_embedding=nbf_ent_rel_embedding)

        return text_ent_rel_embedding

    # def get_target_embedding(self, text_embedding, graph, hr_hidden_states, nbf_rel_output, nbf_query_rel_output,
    #                          nbf_ent_output, h_kg_id, t_index, text_aggregator):
    def get_target_embedding(self,
                             text_embedding,
                             graph,
                             hr_hidden_states,
                             nbf_rel_output,
                             # nbf_query_rel_output,
                             # nbf_ent_output,
                             # hyper_ent_output,
                             h_kg_id,
                             r_kg_id,
                             # t_index,
                             batch,
                             text_aggregator
                             ):
        text_ent_embedding = text_aggregator(
            text_embedding,
            self.entity2orgTokens['input_ids'][h_kg_id],
            # torch.concatenate([self.entity2orgTokens['input_ids'],
            #                    self.relation2orgTokens['input_ids']], dim=0),
            self.entity2orgTokens['attention_mask'][h_kg_id],
            # torch.concatenate([self.entity2orgTokens['attention_mask'],
            #                    self.relation2orgTokens['attention_mask']],
            #                   dim=0),
            # additional_embedding=nbf_ent_output
        )#.unsqueeze(0).repeat(self.bs, 1, 1)
        # text_rel_embedding = text_aggregator(
        #     text_embedding,
        #     self.relation2orgTokens['input_ids'][r_kg_id],
        #     # torch.concatenate([self.entity2orgTokens['input_ids'],
        #     #                    self.relation2orgTokens['input_ids']], dim=0),
        #     self.relation2orgTokens['attention_mask'][r_kg_id],
        #     # torch.concatenate([self.entity2orgTokens['attention_mask'],
        #     #                    self.relation2orgTokens['attention_mask']],
        #     #                   dim=0),
        #     # additional_embedding=nbf_ent_output
        # )#.unsqueeze(0).repeat(self.bs, 1, 1)
        # fusion_weights = F.softmax(
        #     self.tail_weight(
        #         torch.cat([text_ent_rel_embedding,
        #                    torch.cat([nbf_ent_output, nbf_rel_output], dim=1),
        #                    # hyper_ent_output
        #                    ], dim=-1)
        #     ),
        #     dim=-1
        # ).unsqueeze(-1)
        # fusion_ent_rel_embedding = torch.sum(
        #     fusion_weights * torch.cat(
        #         [text_ent_rel_embedding.unsqueeze(2),
        #          torch.cat([nbf_ent_output, nbf_rel_output], dim=1).unsqueeze(2),
        #          # hyper_ent_output.unsqueeze(1)
        #          ],
        #         dim=2
        #     ),
        #     dim=2
        # )
        # logits = torch.matmul(t_hidden_states, hr_hidden_states.unsqueeze(1).transpose(1, 2)).squeeze(-1)
        # inputs = torch.zeros(self.bs, graph.num_nodes, self.args.dim).to(self.device)
        # inputs[torch.arange(self.bs, device=h_kg_id.device), h_kg_id] = text_ent_embedding
        input_ent_representations = torch.zeros(self.bs, graph.num_nodes, self.args.dim).to(self.device)
        # input_rel_representations = torch.zeros(self.bs, graph.num_relations, self.args.dim).to(self.device)
        input_ent_representations[torch.arange(self.bs, device=self.device), h_kg_id] = text_ent_embedding
        # input_rel_representations[torch.arange(self.bs, device=self.device), r_kg_id] = text_rel_embedding

        #data, relation_representations, batch, inputs
        output, t_index = self.tail_context_retriever(
            data=graph,
            batch=batch,
            relation_representations=nbf_rel_output, #text_ent_rel_embedding[:, nbf_ent_output.size(1):, :],
            inputs=input_ent_representations, #text_ent_rel_embedding[:, :nbf_ent_output.size(1), :]
            relation_hyper_flag=True
        )

        hr_hidden_states = hr_hidden_states.unsqueeze(1).repeat(1, t_index.size(1), 1)
        # output1 = torch.cat([nbf_ent_output, hr_hidden_states], dim=-1)
        index = t_index.unsqueeze(-1).expand(-1, -1, output.shape[-1]).to(output.device)
        logits1 = self.mlp(torch.cat([output.gather(1, index), hr_hidden_states], dim=-1)).squeeze(-1)

        return logits1 #target_embedding.reshape(self.bs, self.args.negative_sample_num + 1, -1)

    def forward(
            self,
            batch,
            graph,
            entity2orgTokens,
            relation2orgTokens,
            tokenizer,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        self.entity2orgTokens = entity2orgTokens
        self.relation2orgTokens = relation2orgTokens

        input_ids = batch.input_text.input_ids#.to(self.args.device)
        attention_mask = batch.input_text.attention_mask#.to(self.args.device)
        self.bs = input_ids.size(0)
        # org_vocab_size = tokenizer.vocab_size
        org_token_mask = input_ids < tokenizer.vocab_size
        ent_rel_token_mask = input_ids >= (tokenizer.vocab_size + 2)
        ent_rel_idx = torch.where(input_ids >= (tokenizer.vocab_size + 2))
        ent_rel_ids = input_ids[ent_rel_idx[0], ent_rel_idx[1]] - (tokenizer.vocab_size + 2)
        assert len(ent_rel_ids) % self.bs == 0
        ent_graph_id, rel_graph_id = tokenizer.encode(['<Ent_Graph>', '<Rel_Graph>'], add_special_tokens=False)
        ent_graph_mask = input_ids == ent_graph_id
        rel_graph_mask = input_ids == rel_graph_id
        # ent_graph_idx = torch.where(input_ids == ent_graph_id)
        # rel_graph_idx = torch.where(input_ids == rel_graph_id)
        # query_ent_rel_embedding = ent_rel_embedding[ent_rel_ids]
        h_kg_id = batch.h_kg_id #batch.triplets[:, 0, 0]#batch.h_kg_id
        r_kg_id = batch.r_kg_id #batch.triplets[:, 0, 2]#batch.r_kg_id
        t_kg_id = batch.t_kg_id #batch.triplets[:, 0, 1]#batch.t_kg_id

        ent_rel_embedding = self.get_ent_rel_embedding(self.llm.get_input_embeddings().weight,
                                                       ent_rel_ids,
                                                       torch.concatenate([self.entity2orgTokens['input_ids'],
                                                                          self.relation2orgTokens['input_ids']], dim=0),
                                                       torch.concatenate([self.entity2orgTokens['attention_mask'],
                                                                          self.relation2orgTokens['attention_mask']],
                                                                         dim=0),
                                                       self.text_aggregator_head,
                                                       # nbf_ent_rel_embedding=torch.concatenate([nbf_ent_output, nbf_rel_output], dim=1),
                                                       # hyper_ent_rel_embedding=torch.concatenate([hyper_ent_output, hyper_rel_output], dim=1)
                                                       )
        #input_ent_representations = torch.zeros(self.bs, graph.num_nodes, self.args.dim).to(self.device)
        #input_rel_representations = torch.zeros(self.bs, graph.num_relations, self.args.dim).to(self.device)
        #input_ent_representations[torch.arange(self.bs, device=self.device), h_kg_id] = ent_rel_embedding[h_kg_id]
        #input_rel_representations[torch.arange(self.bs, device=self.device), r_kg_id] = ent_rel_embedding[r_kg_id + graph.num_nodes]
        (nbf_rel_output,
         nbf_query_rel_output,
         nbf_ent_output,
         candidate_nbf_target_output,
         score,
         h_index,
         t_index,
         r_index) = self.kgmodel(data=graph,
                                 batch=batch.triplets,
                                 #relation_representations=input_rel_representations,
                                 #inputs=input_ent_representations
                                 ) # self.kgmodel(batch, graph, h_kg_id, r_kg_id)


        input_embs = torch.zeros(
            *input_ids.shape, self.llm_hidden_dim).to(self.args.device)
        input_embs[org_token_mask] = self.llm.get_input_embeddings()(input_ids[org_token_mask])
        # input_embs[~mask] = self.up_scaling(ent_rel_embedding[ent_rel_idx[0], ent_rel_ids]) # torch.concat([query_ent_embedding, query_rel_embedding], dim=0)[indices].type(input_embs.dtype)
        input_embs[ent_rel_token_mask] = self.up_scaling_ent_rel_token(ent_rel_embedding[ent_rel_ids]) # torch.concat([query_ent_embedding, query_rel_embedding], dim=0)[indices].type(input_embs.dtype)
        input_embs[ent_graph_mask] = self.up_scaling_ent_rel_graph(nbf_ent_output[torch.arange(self.bs), h_kg_id])
        input_embs[rel_graph_mask] = self.up_scaling_ent_rel_graph(nbf_rel_output[torch.arange(self.bs), r_kg_id])

        transformer_outputs = self.llm(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            structure_embeddings=candidate_nbf_target_output,
            # graph=graph,
            # triplets=batch.triplets,
            # # query_nbf_ent_rel_embedding=query_nbf_ent_rel_embedding,
            # ent_rel_idx=ent_rel_idx
        )

        hr_hidden_states = self.down_scaling_hidden(
            transformer_outputs.hidden_states[-1][torch.arange(self.bs, device=self.args.device),
            torch.sum(attention_mask, dim=-1) - 1]
        )
        # h_hidden_states = self.down_scaling_hidden(
        #     transformer_outputs.hidden_states[-1][torch.arange(self.bs, device=self.args.device),
        #     torch.sum(attention_mask, dim=-1) - 2])
        # r_hidden_states = self.down_scaling_hidden(
        #     transformer_outputs.hidden_states[-1][torch.arange(self.bs, device=self.args.device),
        #     torch.sum(attention_mask, dim=-1) - 1])

        logits1 = self.get_target_embedding(
            self.llm.lm_head.weight.data,
            graph,
            hr_hidden_states,
            nbf_rel_output,  # nbf_rel_output, ent_rel_embedding[:, nbf_ent_output.size(1):, :]
            # nbf_query_rel_output,  # nbf_query_rel_output,
            # nbf_ent_output,  #[torch.arange(self.bs), h_kg_id],
            # hyper_ent_output[torch.arange(self.bs), h_kg_id],
            h_kg_id,
            r_kg_id,
            # t_index,
            batch.triplets,
            self.text_aggregator_tail
        )
        # logits = self.get_target_embedding(self.llm.lm_head.weight.data, graph, hr_hidden_states, nbf_rel_output,
        #                                    nbf_query_rel_output,  h_kg_id, t_index, self.text_aggregator_tail)

        if self.training:
            return (
                self.loss(logits1),
                self.loss(score),
                F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(score, dim=-1), reduction='mean'),
                F.kl_div(F.log_softmax(score, dim=-1), F.softmax(logits1, dim=-1), reduction='mean')
            )
        else:
            return (F.softmax(logits1.clamp_min(1e-9), dim=-1) + F.softmax(score.clamp_min(1e-9), dim=-1)) / 2

    def norm(self, x):
        return F.normalize(x, p=2, dim=1)

    def loss(self, logits):
        target = torch.zeros_like(logits)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        neg_weight = torch.ones_like(logits)
        if self.args.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    logits[:, 1:] / self.args.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / self.args.negative_sample_num
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        # if all_loss is not None:
        #     loss = loss + all_loss
        #
        # metric['loss'] = loss

        return loss

    # def
