import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpora_tasks', type=str, default="1p.2p.3p.2i.3i.ip.pi")
    parser.add_argument('--train_tasks', type=str, default="1p.2p.3p.2i.3i")  # 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up
    parser.add_argument('--test_tasks', type=str, default="1p")
    parser.add_argument('--evaluate_union', type=str, default="DNF")
    parser.add_argument('--init_ckpt', type=str, default="./Llama-2-7b-chat-hf_LORA/pf_all_epoch_3/model.pth")  # "./Llama-2-7b-chat-hf_LORA/pf_all_epoch_3/model.pth"
    parser.add_argument('--llm_path', type=str, default="./llm_source")
    parser.add_argument('--save_path', type=str, default="save_model")
    parser.add_argument('--llm_name', type=str, default="Llama-2-7b-chat-hf")  # flan-t5-large  Llama-2-7b-chat-hf
    parser.add_argument('--data_path', type=str, default="./data")
    # WN18RR, CoDEx_M, FB15k237, FB15k237_V1,FB15k237_V2,FB15k237_V3,FB15k237_V4,NELL_V1,NELL_V2,NELL_V3,NELL_V4,WN18RR_V1,WN18RR_V2,WN18RR_V3,WN18RR_V4,FB25,FB50,FB75,FB100,NL0,NL25,NL50,NL75,NL100,WK25,WK50,WK75,WK100
    parser.add_argument('--dataset', type=str, default="FB15k237_V1")  # FB15k-237-betae  Ind-FB106
    parser.add_argument('--trans_or_ind', type=str, default="inductive")  # transductive, inductive, full_inductive
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=0.0005)
    parser.add_argument('--lr_lora', type=int, default=0.0005)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--adversarial_temperature', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--kd_lambda', type=float, default=0.5)
    parser.add_argument('--atte_lambda', type=float, default=0.01)
    parser.add_argument('--structure_fq', type=int, default=4)
    parser.add_argument('--valid_steps', type=int, default=10000)



    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--chat_lm', type=str, default="gpt", choices=['gpt', 'llama'])
    parser.add_argument('--embed_lm', type=str, default="gpt", choices=['gpt', 'local'])
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--local_embed_lm', type=str, default="./embedding_lm/all-MiniLM-L12-v2")
    parser.add_argument('--beam_num', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.0002)
    parser.add_argument('--aggregate_func_nbf', type=str, default='sum')
    parser.add_argument('--aggregate_func_text', type=str, default='pna')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--negative_sample_num', type=int, default=256)
    return parser.parse_args()
