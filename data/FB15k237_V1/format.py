import os
import pickle

def map_ent_rel():
    train_file_names = ['train.txt', 'valid.txt', 'test.txt']
    test_file_names = ['train_ind.txt', 'valid_ind.txt', 'test_ind.txt']
    with open(os.path.join('org_kg', f'train.txt'), 'r', encoding='utf-8') as f:
        lines_train = f.readlines()
    with open(os.path.join('org_kg', f'valid.txt'), 'r', encoding='utf-8') as f:
        lines_train += f.readlines()
    # with open(os.path.join('org_kg', f'test.txt'), 'r', encoding='utf-8') as f:
    #     lines_train += f.readlines()

    with open(os.path.join('org_kg', f'train_ind.txt'), 'r', encoding='utf-8') as f:
        lines_test = f.readlines()
    with open(os.path.join('org_kg', f'valid_ind.txt'), 'r', encoding='utf-8') as f:
        lines_test += f.readlines()
    with open(os.path.join('org_kg', f'test_ind.txt'), 'r', encoding='utf-8') as f:
        lines_test += f.readlines()

    num_ent_train = 0
    num_ent_test = 0
    num_rel = 0
    ent2id_train = {}
    ent2id_test = {}
    rel2id = {}

    for line in lines_train:
        h, r, t = line.strip().split('\t')
        pos_r = '+' + r
        # inver_r = '-' + r
        if h not in ent2id_train:
            ent2id_train[h] = num_ent_train
            num_ent_train += 1
        if t not in ent2id_train:
            ent2id_train[t] = num_ent_train
            num_ent_train += 1
        if pos_r not in rel2id:
            rel2id[pos_r] = num_rel
            num_rel += 1

    for line in lines_test:
        h, r, t = line.strip().split('\t')
        pos_r = '+' + r
        # inver_r = '-' + r
        if h not in ent2id_test:
            ent2id_test[h] = num_ent_test
            num_ent_test += 1
        if t not in ent2id_test:
            ent2id_test[t] = num_ent_test
            num_ent_test += 1
        if pos_r not in rel2id:
            rel2id[pos_r] = num_rel
            num_rel += 1

    inverse_rel2id = {}
    for k, v in rel2id.items():
        inverse_rel2id[k.replace('+', '-')] = v + num_rel
    rel2id.update(inverse_rel2id)
    id2ent_train = {v: k for k, v in ent2id_train.items()}
    id2ent_test = {v: k for k, v in ent2id_test.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    print()
    with open(f'ent2id.pkl', 'wb') as f:
        pickle.dump(ent2id_train, f)
    with open(f'rel2id.pkl', 'wb') as f:
        pickle.dump(rel2id, f)
    with open(f'id2ent.pkl', 'wb') as f:
        pickle.dump(id2ent_train, f)
    with open(f'id2rel.pkl', 'wb') as f:
        pickle.dump(id2rel, f)

    with open(f'ent2id_ind.pkl', 'wb') as f:
        pickle.dump(ent2id_test, f)
    with open(f'rel2id_ind.pkl', 'wb') as f:
        pickle.dump(rel2id, f)
    with open(f'id2ent_ind.pkl', 'wb') as f:
        pickle.dump(id2ent_test, f)
    with open(f'id2rel_ind.pkl', 'wb') as f:
        pickle.dump(id2rel, f)

    with open(os.path.join(f'stats.txt'), 'w', encoding='utf-8') as f:
        f.writelines(f'numentity: {num_ent_train}\nnumrelations: {len(rel2id)}')
    with open(os.path.join(f'stats_ind.txt'), 'w', encoding='utf-8') as f:
        f.writelines(f'numentity: {num_ent_test}\nnumrelations: {len(rel2id)}')
    # return ent2id, id2ent, rel2id, id2rel

def get_reverse_triplets(org_kg, ent2id, rel2id):
    new_kg = []
    for triplet in org_kg:
        h, r, t = triplet.strip().split('\t')
        pos_r = '+' + r
        new_kg.append(f"{ent2id[h]}\t{rel2id[pos_r]}\t{ent2id[t]}\n")
    for triplet in org_kg:
        h, r, t = triplet.strip().split('\t')
        inver_r = '-' + r
        new_kg.append(f"{ent2id[t]}\t{rel2id[inver_r]}\t{ent2id[h]}\n")
    return new_kg

def get_full_kg():

    with open(os.path.join('org_kg', f'train.txt'), 'r', encoding='utf-8') as f:
        org_train_kg = f.readlines()
    with open(os.path.join('org_kg', f'valid.txt'), 'r', encoding='utf-8') as f:
        org_valid_kg = f.readlines()
    # with open(os.path.join('org_kg', f'test.txt'), 'r', encoding='utf-8') as f:
    #     org_test_kg = f.readlines()
    with open(f'ent2id.pkl', 'rb') as f:
        ent2id = pickle.load(f)
    with open(f'rel2id.pkl', 'rb') as f:
        rel2id = pickle.load(f)

    new_train_kg = get_reverse_triplets(org_train_kg, ent2id, rel2id)
    new_valid_kg = get_reverse_triplets(org_valid_kg, ent2id, rel2id)
    # new_test_kg = get_reverse_triplets(org_test_kg, ent2id, rel2id)

    with open(os.path.join(f'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(''.join(new_train_kg))
    with open(os.path.join(f'valid.txt'), 'w', encoding='utf-8') as f:
        f.writelines(''.join(new_valid_kg))
    # with open(os.path.join(f'test.txt'), 'w', encoding='utf-8') as f:
    #     f.writelines(''.join(new_test_kg))
    ##############################################################################
    with open(os.path.join('org_kg', f'train_ind.txt'), 'r', encoding='utf-8') as f:
        org_train_kg_ind = f.readlines()
    with open(os.path.join('org_kg', f'valid_ind.txt'), 'r', encoding='utf-8') as f:
        org_valid_kg_ind = f.readlines()
    with open(os.path.join('org_kg', f'test_ind.txt'), 'r', encoding='utf-8') as f:
        org_test_kg_ind = f.readlines()
    with open(f'ent2id_ind.pkl', 'rb') as f:
        ent2id_ind = pickle.load(f)
    with open(f'rel2id_ind.pkl', 'rb') as f:
        rel2id_ind = pickle.load(f)

    new_train_kg_ind = get_reverse_triplets(org_train_kg_ind, ent2id_ind, rel2id_ind)
    new_valid_kg_ind = get_reverse_triplets(org_valid_kg_ind, ent2id_ind, rel2id_ind)
    new_test_kg_ind = get_reverse_triplets(org_test_kg_ind, ent2id_ind, rel2id_ind)

    with open(os.path.join(f'train_ind.txt'), 'w', encoding='utf-8') as f:
        f.writelines(''.join(new_train_kg_ind))
    with open(os.path.join(f'valid_ind.txt'), 'w', encoding='utf-8') as f:
        f.writelines(''.join(new_valid_kg_ind))
    with open(os.path.join(f'test_ind.txt'), 'w', encoding='utf-8') as f:
        f.writelines(''.join(new_test_kg_ind))

    print()



map_ent_rel()
get_full_kg()
