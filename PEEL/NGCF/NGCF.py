'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import config_all

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        #self.max_num_blocks_per = args.num_blocks_per

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'loss_weight': nn.Parameter(initializer(torch.empty(1,1))),
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):

        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_rule_loss(self, users, pos_items, neg_items, num_blocks_per):
        #

        # pos_scores = torch.sum(torch.einsum('bi,dj->bdij', (users, pos_items)), (1,2))
        # neg_scores = torch.sum(torch.einsum('bi,dj->bdij', (users, neg_items)), (1,2))
        #
        # maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        #
        # mf_loss = -1 * torch.mean(maxi)

        len_block = pos_items.shape[1]//num_blocks_per
        len_block_u = users.shape[1]//num_blocks_per

        splits = torch.split(pos_items, len_block, dim=1)
        splits_n = torch.split(neg_items, len_block, dim=1)
        # splits_u = torch.split(users, len_block_u, dim=1)

        #reshaped_users = torch.reshape(users, (users.shape[0], 1, users.shape[1], 1))
        
        sum_pos = 0
        sum_neg = 0
        for block_p, block_n in zip(splits, splits_n):

            block_tiled_p = block_p.repeat(1, num_blocks_per)
            block_tiled_sum_p = torch.sum(torch.mul(users, block_tiled_p), axis=1)

            #reshaped_p = torch.reshape(block_p, (block_p.shape[0], 1, block_p.shape[1]))    
            #block_tiled_sum_p = torch.sum(torch.matmul(reshaped_users, reshaped_p), dim = (2,3))

            sum_pos += block_tiled_sum_p

            block_tiled_n = block_n.repeat(1, num_blocks_per)
            block_tiled_sum_n = torch.sum(torch.mul(users, block_tiled_n), axis=1)
            
            #reshaped_n = torch.reshape(block_p, (block_p.shape[0], 1, block_p.shape[1]))
            #block_tiled_sum_n = torch.sum(torch.matmul(reshaped_users, reshaped_n), dim = (2,3))
            sum_neg += block_tiled_sum_n

        maxi = nn.LogSigmoid()(sum_pos - sum_neg)

        mf_loss = -1 * torch.mean(maxi)

        #regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2

        div_loss = 0
        #sum_norms = 0
        #for i in range(num_blocks_per):
        #    if i < num_blocks_per - 1:
        #        sum_norms += (torch.norm(splits[i] - splits[i+1]) ** 2)
        #div_loss = -1 * self.decay * sum_norms / self.batch_size
        #div_loss = -0.0001 * sum_norms

        return mf_loss + div_loss, mf_loss, div_loss
        #return mf_loss + div_loss + 0.0001*regularizer, mf_loss, div_loss

    # def rating(self, u_g_embeddings, pos_i_g_embeddings):
    #     return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def rating(self, u_g_embeddings, pos_i_g_embeddings):

        #num_blocks_per = 8 # todo
        num_blocks_per = config_all.max_num_blocks_per

        pos_items = pos_i_g_embeddings
        users = u_g_embeddings

        len_block = pos_items.shape[1]//num_blocks_per
        # len_block_u = users.shape[1]//num_blocks_per

        splits = torch.split(pos_items, len_block, dim=1)
        # splits_u = torch.split(users, len_block_u, dim=1)


        sum_pos = torch.zeros((u_g_embeddings.shape[0], pos_i_g_embeddings.shape[0])).to(self.device)
        #
        # ipdb.set_trace()
        for block_p in splits:
            
            block_tiled_p = block_p.repeat(1, num_blocks_per)
            block_tiled_sum_p = torch.matmul(users, block_tiled_p.t())
                        
            '''
            reshaped_users = torch.reshape(users, (u_g_embeddings.shape[0], 1, u_g_embeddings.shape[1], 1))
            reshaped_p = torch.reshape(block_p, (block_p.shape[0], 1, block_p.shape[1]))
            block_tiled_sum_p = torch.sum(torch.matmul(reshaped_users, reshaped_p), dim = (2,3))
            '''
            sum_pos += block_tiled_sum_p

        # ipdb.set_trace()
        #
        # # ipdb.set_trace()
        # sum = torch.zeros(sum_pos[0].shape).to(self.device)
        # for i in range(len(sum_pos)):
        #     sum += sum_pos[i].to(self.device)
        # del sum_pos

        return sum_pos

        # return (torch.sum(torch.stack(sum_pos, dim=0), 0))

        # return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # # transformed sum messages of neighbors.
            # sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
            #                                  + self.weight_dict['b_gc_%d' % k]
            #
            # # bi messages of neighbors.
            # # element-wise product
            # bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # # transformed bi messages of neighbors.
            # bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
            #                                 + self.weight_dict['b_bi_%d' % k]
            #
            # # non-linear activation.
            # ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            #
            # # message dropout.
            # ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            #
            #
            # # normalize the distribution of embeddings.
            # norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        #all_embeddings = ego_embeddings
        #all_embeddings = torch.mean(torch.stack(all_embeddings, 0), dim=0)
        all_embeddings = torch.cat(all_embeddings, 1)

        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """

        #

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
