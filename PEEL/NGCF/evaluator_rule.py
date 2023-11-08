import numpy as np
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import torch
# import ipdb
import json
import torch.nn as nn
import heapq

import config_all
import os

args = parse_args()
args.device = torch.device('cuda:' + str(args.gpu_id))
#args.device = torch.device(1)
# ipdb.set_trace()
print(args.device)
Ks = eval(args.Ks)
K = config_all.K  # num of combinations
max_num_blocks_per = config_all.max_num_blocks_per  # the max num of blocks per item
G = config_all.G  # num of groups
M = config_all.M  # memory upperbound (converted to the max num of total blocks)
BATCH_SIZE = config_all.BATCH_SIZE


cores = multiprocessing.cpu_count() // 2

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
ITEM_NUM = data_generator.n_items
USER_NUM = data_generator.n_users

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}



def rating(u_g_embeddings, pos_i_g_embeddings, max_num_blocks_per):


    pos_items = pos_i_g_embeddings
    users = u_g_embeddings

    len_block = pos_items.shape[1]//num_blocks_per
    # len_block_u = users.shape[1]//num_blocks_per

    splits = torch.split(pos_items, len_block, dim=1)
    # splits_u = torch.split(users, len_block_u, dim=1)


    sum_pos = []
    #

    for block_p in splits:
        # block_tiled_p = torch.tile(block_p, (1, num_blocks_per))
        block_tiled_p = block_p.repeat(1, num_blocks_per)
        block_tiled_sum_p = torch.matmul(users, block_tiled_p.t())
        sum_pos.append(block_tiled_sum_p)
    #

    return (torch.sum(torch.stack(sum_pos, dim=0), 0)*max_num_blocks_per/num_blocks_per)


def rating_eva(u_g_embeddings, pos_i_g_embeddings, max_num_blocks_per, combination, G):

    pos_items = pos_i_g_embeddings
    len_group = (pos_items.shape[0] // G) + 1
    pos_items_grouped = torch.split(pos_items, len_group, dim=0) #split item embeddings into G groups
    len_block = pos_items.shape[1] // max_num_blocks_per
    users = u_g_embeddings

    idx = 0
    g_scores = []
    #
    #print(np.count_nonzero(combination)-20)
    for g in combination:
        b_multi_hot = g[G:]
        b_multi_hot_idx = [i for i, j in enumerate(b_multi_hot) if j == 1]

        #b_multi_hot_idx = [i for i in range(max_num_blocks_per)]
        #print(b_multi_hot_idx)

        ## for cuda cpu consistency
        temp_split = torch.split(pos_items_grouped[idx], len_block, dim=1)
        temp_split_cpu = tuple(temp.cpu() for temp in temp_split)
        splits = np.array(temp_split_cpu)[b_multi_hot_idx]

        sum_pos = []
        # print('g', g)
        # print(splits[0].shape)
        # if idx >= 3:
        #

        for block_p in splits:
            block_tiled_p = block_p.cuda().repeat(1, max_num_blocks_per)
            block_tiled_sum_p = torch.matmul(users, block_tiled_p.t())
            '''
            reshaped_users = torch.reshape(users, (u_g_embeddings.shape[0], 1, u_g_embeddings.shape[1], 1))
            reshaped_p = torch.reshape(block_p, (block_p.shape[0], 1, block_p.shape[1]))
            block_tiled_sum_p = torch.sum(torch.matmul(reshaped_users, reshaped_p), dim = (2,3))
            '''
            sum_pos.append(block_tiled_sum_p)
            #print(block_tiled_sum_p.shape)
        #
        #print(torch.stack(sum_pos, dim=0).shape)
        g_scores.append(torch.sum(torch.stack(sum_pos, dim=0), 0)*max_num_blocks_per/len(b_multi_hot_idx))
        idx += 1
    #

    return torch.cat(g_scores, dim=1)


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    #print(len(test_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)



def test(users_to_test, u_g_embeddings, pos_i_g_embeddings, max_num_blocks_per, G, combination):


    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)
    u_batch_size = BATCH_SIZE * 2

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        #print(user_batch)

        # all-item test
        u_g_embeddings_batch = u_g_embeddings[user_batch]
        rate_batch = rating_eva(u_g_embeddings_batch, pos_i_g_embeddings, max_num_blocks_per, combination, G).detach().to(args.device)
        #print(rate_batch)

        user_batch_rating_uid = zip(rate_batch.cpu().numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result



def evaluator_rule_main(combinations, user_embs, item_embs):


    item_emb_weight = torch.FloatTensor(item_embs).cuda()
    # item_embedding = nn.Embedding.from_pretrained(item_emb_weight)

    user_emb_weight = torch.FloatTensor(user_embs).cuda()
    # user_embedding = nn.Embedding.from_pretrained(user_emb_weight)

    users_to_test = list(data_generator.test_set.keys())

    item_groups = torch.split(item_emb_weight, G, dim=0)

    recall_100_all = []

    for c in combinations:
        print('hi')
        result = test(users_to_test, user_emb_weight, item_emb_weight, max_num_blocks_per, G, c)
        recall_100_all.append(result['recall'][-1])

    recalls = np.array(recall_100_all)
    np.save("recall_100"+"_G"+str(G)+"_M"+str(M)+"_NP"+str(max_num_blocks_per)+"_K"+str(K)+".npy", recalls)

    return recalls


#
if __name__ == '__main__':

    combinations = np.load("combinations"+"_G"+str(G)+"_M"+str(M)+"_NP"+str(max_num_blocks_per)+"_K"+str(K)+".npy")
    item_embs = np.load('item_emb.npy') # full trained emb
    user_embs = np.load('user_emb.npy') # full trained emb

    item_emb_weight = torch.FloatTensor(item_embs).cuda()
    # item_embedding = nn.Embedding.from_pretrained(item_emb_weight)

    user_emb_weight = torch.FloatTensor(user_embs).cuda()
    # user_embedding = nn.Embedding.from_pretrained(user_emb_weight)

    users_to_test = sorted(list(data_generator.test_set.keys()))[:100]

    item_groups = torch.split(item_emb_weight, G, dim=0)

    recall_100_all = []

    i = 1
    for c in combinations:
        print(i)
        i += 1
        #print('hi')
        result = test(users_to_test, user_emb_weight, item_emb_weight, max_num_blocks_per, G, c)
        #print(result['recall'])
        recall_100_all.append(result['recall'][-1])

    np.save("recall_100"+"_G"+str(G)+"_M"+str(M)+"_NP"+str(max_num_blocks_per)+"_K"+str(K)+".npy", np.array(recall_100_all))



    print('end')

