'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time

import numpy as np

import ipdb

#from mtadam import MTAdam
from sklearn.cluster import KMeans

if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))
    #args.device = torch.device(0)

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = MTAdam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()

            #
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            # batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
            #                                                                   pos_i_g_embeddings,
            #                                                                   neg_i_g_embeddings)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_rule_loss(u_g_embeddings,
                                                                               pos_i_g_embeddings,
                                                                               neg_i_g_embeddings,
                                                                               args.num_blocks_per)


            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            #print(batch_loss)
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if epoch % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        #users_to_test = list(data_generator.test_set.keys())[:1000]
        users_to_test = sorted(list(data_generator.test_set.keys()))[:1000]

        ret = test(model, users_to_test, drop_flag=False, batch_test_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                        ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3],ret['recall'][4],
                        ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3],ret['precision'][4],
                        ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][3],ret['hit_ratio'][4],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4]
             )
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][-1], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][-1] == cur_best_pre_0 and args.save_flag == 1:        
            i_batch_size = args.batch_size
            n_item_batchs = data_generator.n_items // i_batch_size + 1
            item_forwarded_emb = np.zeros(shape=(data_generator.n_items, args.embed_size*4))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, data_generator.n_items)

                item_batch = range(i_start, i_end)

                _, i_g_embeddings, _ = model([],
                                              item_batch,
                                              [],
                                              drop_flag=False)
                item_forwarded_emb[i_start:i_end, :] = i_g_embeddings.detach().cpu().numpy()

            np.save('item_emb_cat.npy', item_forwarded_emb)
            #np.save('item_emb.npy', model.state_dict()['embedding_dict.item_emb'].cpu().numpy())

            u_batch_size = args.batch_size
            n_user_batchs = data_generator.n_users // u_batch_size + 1
            user_forwarded_emb = np.zeros(shape=(data_generator.n_users, args.embed_size*4))

            u_count = 0
            for u_batch_id in range(n_user_batchs):
                u_start = u_batch_id * u_batch_size
                u_end = min((u_batch_id + 1) * u_batch_size, data_generator.n_users)

                user_batch = range(u_start, u_end)

                u_g_embeddings, _, _ = model(user_batch,
                                              [],
                                              [],
                                              drop_flag=False)
                user_forwarded_emb[u_start:u_end, :] = u_g_embeddings.detach().cpu().numpy()
            
            np.save('user_emb_cat.npy', user_forwarded_emb)
            #np.save('user_emb.npy', model.state_dict()['embedding_dict.user_emb'].cpu().numpy())

            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    #
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)