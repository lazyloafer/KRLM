import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import transformers
from functools import partial
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    TaskType
    # prepare_model_for_int8_training,
)
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from llama_model import LlamaForKGReasoning
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration
from configs import train_args
from loader import LQLDatasetTrain, LQLDatasetValid, collate_train, collate_test, multigraph_collator
from model_v2 import LQLModel #, Ultra
from tasks import get_nb_trainable_parameters, strict_negative_mask, compute_ranking, load_ckpt
from motif import util
from motif.models import MOTIF
import time
import warnings
warnings.filterwarnings('ignore')
args = train_args()
config = ProjectConfiguration(project_dir=".", logging_dir="logs")
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="tensorboard", project_config=config)
accelerator.init_trackers(time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())))
# state = torch.load('./Llama-2-7b-chat-hf_LORA/epoch_9/model.pth', map_location='cpu')
dataset_name = args.dataset
data_train_list = []
tokenizer_train_list = []
data_loader_train_list = []
data_test_list = []
tokenizer_test_list = []
filtered_data_list = []
data_loader_test_list = []
total_test_num_list = [0]
for dataset in tqdm(dataset_name.split(',')):
    args.dataset = dataset
    data_train = LQLDatasetTrain(args=args, mode='train')
    tokenizer_train = data_train.tokenizer_train
    data_loader_train = DataLoader(data_train, batch_size=args.train_batch_size, pin_memory=True, num_workers=args.num_workers,
                                   collate_fn=partial(collate_train, args=args, graph=data_train.train_graph, tokenizer=tokenizer_train), shuffle=True)

    data_test = LQLDatasetValid(args=args, mode='test', test_fast=False)
    tokenizer_test = data_test.tokenizer_test
    filtered_data = data_test.filtered_data.to(args.device)
    data_loader_test = DataLoader(data_test, batch_size=args.test_batch_size, pin_memory=True, num_workers=args.num_workers,
                                   collate_fn=partial(collate_test, args=args, graph=data_test.test_graph, tokenizer=tokenizer_test), shuffle=False)

    data_train_list.append(data_train)
    tokenizer_train_list.append(tokenizer_train)
    data_loader_train_list.append(data_loader_train)

    data_test_list.append(data_test)
    tokenizer_test_list.append(tokenizer_test)
    filtered_data_list.append(filtered_data)
    data_loader_test_list.append(data_loader_test)
    total_test_num_list.append(len(data_test) * 2)

llm = LlamaForKGReasoning.from_pretrained(
    os.path.join(args.llm_path, args.llm_name),
    low_cpu_mem_usage=False,
    dim=args.dim,
    # aggregate_func_nbf=args.aggregate_func_nbf,
    # num_relation=len(data_train.relTokens)
    # # use_flash_attention_2=True, # leading to an error
)
# llm.resize_token_embeddings(len(tokenizer))
peft_config = LoraConfig(
    r=32,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'],  # 'q_proj', 'k_proj', 'v_proj'
    bias='none',
    task_type=TaskType.CAUSAL_LM  # CAUSAL_LM  SEQ_2_SEQ_LM
)
for name, param in llm.named_parameters(): # 26,291,138
    if 'structure' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


motif_args, motif_vars = util.parse_args() # args.trans_or_ind
mofit_cfg = util.load_config(motif_args.config, context=motif_vars)



kgmodel = MOTIF(
        rel_model_cfg=mofit_cfg.model.relation_model,
        entity_model_cfg=mofit_cfg.model.entity_model,
    )
state = torch.load('./kgmodel_source/kgfm.pth', map_location='cpu')
kgmodel.load_state_dict(state['model'])
# for name, param in kgmodel.named_parameters(): # 26,291,138
#     param.requires_grad = False

model = LQLModel(args=args, mofit_cfg=mofit_cfg, llm=llm, kgmodel=kgmodel)
# model = LQLModel(args=args, mofit_cfg=mofit_cfg, kgmodel=kgmodel)
model = load_ckpt(args, model)
model = model.to(args.device)

trainable_params, all_param = get_nb_trainable_parameters(model)
accelerator.print(model)
accelerator.print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

num_training_steps = len(data_loader_train) * args.epochs
lora_params = []
extra_params = []
for name, param in model.named_parameters():
    if ('llm' in name) and ('merge' not in name):
        lora_params.append(param)
    else:
        extra_params.append(param)
# optimzer = torch.optim.AdamW(model.parameters(), lr=args.lr)
optimzer = torch.optim.AdamW([{"params": lora_params, "lr": args.lr_lora}, {"params": extra_params, "lr": args.lr}])
lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimzer,
    num_warmup_steps=1#2000, #int(num_training_steps * args.warmup_rate),
    # num_training_steps=num_training_steps
)
model, \
optimzer, \
lr_scheduler = accelerator.prepare(model, optimzer, lr_scheduler)
# data_loader_train_list = [iter(accelerator.prepare(data_loader_train_list[i])) for i in range(len(data_loader_train_list))]
data_loader_test_list = [accelerator.prepare(data_loader_test_list[i]) for i in range(len(data_loader_test_list))]
dataset_name = dataset_name.split(',')

def evaluate(data_loader, model, data_test, filtered_data, tokenizer_test, graph):

    # total_num = 0 # len(data_loader.dataset.data)
    # total_hit1, total_hit3, total_hit10, total_mrr = 0, 0, 0, 0
    total_hit1, total_hit3, total_hit10, total_mrr = [], [], [], []
    with torch.no_grad():
        total_num = 0

        rankings = []
        # num_negatives = []
        # tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
        for triplets, t_batch, h_batch in tqdm(data_loader):
            t_pred = model(batch=t_batch, graph=graph,
                           entity2orgTokens=data_test.entity2orgTokens_test,
                           relation2orgTokens=data_test.relation2orgTokens_test,
                           tokenizer=tokenizer_test, output_hidden_states=True)

            h_pred = model(batch=h_batch, graph=graph,
                           entity2orgTokens=data_test.entity2orgTokens_test,
                           relation2orgTokens=data_test.relation2orgTokens_test,
                           tokenizer=tokenizer_test, output_hidden_states=True)
            t_mask, h_mask = strict_negative_mask(filtered_data, triplets)
            pos_h_index, pos_t_index, pos_r_index = triplets.t()
            t_ranking = compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = compute_ranking(h_pred, pos_h_index, h_mask)

            # rankings += t_ranking.cpu().tolist()
            # rankings += h_ranking.cpu().tolist()
            rankings += [t_ranking, h_ranking]

        rankings = torch.cat(rankings)
        hit1 = (rankings <= 1).cpu().tolist()
        hit3 = (rankings <= 3).cpu().tolist()
        hit10 = (rankings <= 10).cpu().tolist()
        mrr = (1.0 / rankings).cpu().tolist()
        #
        total_hit1.extend(hit1)
        total_hit3.extend(hit3)
        total_hit10.extend(hit10)
        total_mrr.extend(mrr)

        # accelerator.print(f"mean hit1 {total_hit1 / total_num}\nmean hit3 {total_hit3 / total_num}\nmean hit10 {total_hit10 / total_num}\nmean mrr {total_mrr / total_num}")
    return total_hit1, total_hit3, total_hit10, total_mrr

# model.eval()
#
# for i in range(len(dataset_name)):
#     data_loader = data_loader_test_list[i]
#     data_test = data_test_list[i]
#     filtered_data = filtered_data_list[i]
#     tokenizer_test = data_test.tokenizer_test
#     graph = data_test.test_graph.to(args.device)
#     dataset = dataset_name[i]
#     (total_hit1_per_device,
#      total_hit3_per_device,
#      total_hit10_per_device,
#      total_mrr_per_device) = evaluate(data_loader, model, data_test, filtered_data, tokenizer_test, graph)
#     total_hit1, total_hit3, total_hit10, total_mrr = [], [], [], []
#     total_hit1.extend(accelerator.gather_for_metrics(total_hit1_per_device))
#     total_hit3.extend(accelerator.gather_for_metrics(total_hit3_per_device))
#     total_hit10.extend(accelerator.gather_for_metrics(total_hit10_per_device))
#     total_mrr.extend(accelerator.gather_for_metrics(total_mrr_per_device))
#     accelerator.print(
#         f"{dataset} Mean Hit1 {np.mean(total_hit1)}\n"
#         f"{dataset} Mean Hit3 {np.mean(total_hit3)}\n"
#         f"{dataset} Mean Hit10 {np.mean(total_hit10)}\n"
#         f"{dataset} Mean MRR {np.mean(total_mrr)}"
#     )
#     accelerator.log({f"{dataset} Mean Hit1": np.mean(total_hit1)}, step=-1)
#     accelerator.log({f"{dataset} Mean Hit3": np.mean(total_hit3)}, step=-1)
#     accelerator.log({f"{dataset} Mean Hit10": np.mean(total_hit10)}, step=-1)
#     accelerator.log({f"{dataset} Mean MRR": np.mean(total_mrr)}, step=-1)
total_step = 0
train_triplets = torch.cat([
        torch.cat([data_train.test_graph.edge_index, data_train.test_graph.edge_type.unsqueeze(0)]).t()
        for data_train in data_train_list
])
train_loader = DataLoader(train_triplets,
                          args.train_batch_size,
                          collate_fn=partial(multigraph_collator,
                                             data_train_list=data_train_list,
                                             args=args,
                                             tokenizer_train_list=tokenizer_train_list
                                             )
                          )
train_loader = accelerator.prepare(train_loader)
step = 0
for epoch in range(args.epochs):
    # epoch = int(step // args.valid_steps)
    accelerator.print(f"Training on Epoch {epoch}......")
    model.train()
    total_loss = 0
    # step = 0
    total_hit1, total_hit3, total_hit10, total_mrr = [], [], [], []
    for batch, graph, tokenizer, entity2orgTokens_train, relation2orgTokens_train in tqdm(train_loader):
        with accelerator.accumulate(model):  # loss1, loss2, kl1, kl2
            loss1, loss2, kl1, kl2 = model(batch=batch, graph=graph.to(args.device),
                                           entity2orgTokens=entity2orgTokens_train,
                                           relation2orgTokens=relation2orgTokens_train,
                                           tokenizer=tokenizer, output_hidden_states=True)
            loss = (1 - args.kd_lambda) * loss1 + args.kd_lambda * kl1 + (
                        1 - args.kd_lambda) * loss2 + args.kd_lambda * kl2
            with torch.autograd.set_detect_anomaly(True):
                accelerator.backward(loss)
                optimzer.step()
                lr_scheduler.step()
                model.zero_grad()
            total_loss += loss.item()

            accelerator.log({"main_loss": loss.item()}, step=step)
            accelerator.log({"loss1": loss1.item()}, step=step)
            accelerator.log({"loss2": loss2.item()}, step=step)
            # accelerator.log({"loss_nbf": 0.5 * loss_nbf.item()}, step=total_step)
            accelerator.log({"kl1": kl1.item()}, step=step)
            accelerator.log({"kl2": kl2.item()}, step=step)


            if step % 500 == 0:
                accelerator.print(f"Epoch {epoch}-Step {step - epoch * args.valid_steps}: "
                                  f"main_loss: {loss.item()}, "
                                  f"loss1: {loss1.item()}, "
                                  f"loss2: {loss2.item()}, "
                                  f"kl1: {kl1.item()}, "
                                  f"kl2: {kl2.item()}"
                                  )
            step += 1
            # total_step += 1
        if step % args.valid_steps == 0:
            train_ppl = total_loss / step
            accelerator.print(f"Epoch {epoch}: Train_ppl: {train_ppl}")
            accelerator.print(f"Evaluating on Epoch {epoch}......")
            model.eval()
            for i in range(len(dataset_name)):
                data_loader = data_loader_test_list[i]
                data_test = data_test_list[i]
                filtered_data = filtered_data_list[i]
                tokenizer_test = data_test.tokenizer_test
                graph = data_test.test_graph.to(args.device)
                dataset = dataset_name[i]
                (total_hit1_per_device,
                 total_hit3_per_device,
                 total_hit10_per_device,
                 total_mrr_per_device) = evaluate(data_loader, model, data_test, filtered_data, tokenizer_test, graph)
                total_hit1, total_hit3, total_hit10, total_mrr = [], [], [], []
                total_hit1.extend(accelerator.gather_for_metrics(total_hit1_per_device))
                total_hit3.extend(accelerator.gather_for_metrics(total_hit3_per_device))
                total_hit10.extend(accelerator.gather_for_metrics(total_hit10_per_device))
                total_mrr.extend(accelerator.gather_for_metrics(total_mrr_per_device))
                accelerator.print(
                    f"{dataset} Mean Hit1 {np.mean(total_hit1)}\n"
                    f"{dataset} Mean Hit3 {np.mean(total_hit3)}\n"
                    f"{dataset} Mean Hit10 {np.mean(total_hit10)}\n"
                    f"{dataset} Mean MRR {np.mean(total_mrr)}"
                )
                accelerator.log(
                    {f"{dataset} Mean Hit1": np.mean(total_hit1)}, step=epoch)
                accelerator.log(
                    {f"{dataset} Mean Hit3": np.mean(total_hit3)}, step=epoch)
                accelerator.log(
                    {f"{dataset} Mean Hit10": np.mean(total_hit10)}, step=epoch)
                accelerator.log({f"{dataset} Mean MRR": np.mean(total_mrr)}, step=epoch)

            accelerator.print(f"Saving model on Epoch {epoch}......")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # model = accelerator.unwrap_model(model)
                peft_id = f"{args.llm_name}_{peft_config.peft_type}"
                if not os.path.exists(peft_id):
                    os.mkdir(os.path.join(peft_id))
                # model.module.llm.save_pretrained(
                #     save_directory=os.path.join(peft_id, f"epoch_{epoch}"),
                #     state_dict=accelerator.get_state_dict(model))
                #
                extra_model_dict = OrderedDict()
                a = extra_model_dict.keys()
                for name, param in model.state_dict().items():
                    if 'structure' in name:
                        if name not in extra_model_dict:
                            extra_model_dict[name] = param
                    if 'llm' not in name:
                        if name not in extra_model_dict:
                            extra_model_dict[name] = param
                # torch.save(extra_model_dict, os.path.join(peft_id, f"epoch_{epoch}", f"extra_model.pth"))
                if not os.path.exists(os.path.join(peft_id, f"epoch_{epoch}")):
                    os.mkdir(os.path.join(peft_id, f"epoch_{epoch}"))
                torch.save(extra_model_dict, os.path.join(peft_id, f"epoch_{epoch}", f"model.pth"))
            accelerator.wait_for_everyone()
            break

