import numpy as np
import argparse
import utility.metrics as metrics
# from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import torch
from evaluator_rule import *
# import ipdb
import json
import torch.nn as nn
import heapq
# import nfm.config
import nfm.model
import nfm.evaluate
import ipdb
import nfm.data_utils
import sys
import random
import config_all


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', nargs='?', default='../Data/',
					help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='amazon-book',
					help='Choose a dataset from {gowalla, yelp2018, amazon-book, amazon-book-sample}')

parser.add_argument("--dropout",
	default='[0.0, 0.0]',
	help="dropout rate for FM and MLP")
parser.add_argument("--batch_size",
	type=int,
	default=512,
	help="batch size for training")
parser.add_argument("--epochs",
	type=int,
	default=2,
	help="training epochs")
parser.add_argument("--hidden_factor",
	type=int,
	default=36,
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
	default='[64]',
	help="size of layers in MLP model, '[]' is NFM-0")
parser.add_argument("--lamda",
	type=float,
	default=0.0,
	help="regularizer for bilinear layers")
parser.add_argument("--batch_norm",
	default=True,
	help="use batch_norm or not")
parser.add_argument("--pre_train",
	action='store_true',
	default=True,
	help="whether use the pre-train or not")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="1",
	help="gpu card ID")
args = parser.parse_args()

P = config_all.P
C = config_all.C
S = config_all.S

max_num_blocks_per = config_all.max_num_blocks_per  # the max num of blocks per item
G = config_all.G  # num of groups
M = config_all.M  # memory upperbound (converted to the max num of total blocks)

if __name__ == '__main__':

	dir_estimator = config_all.NFM_model
	item_embs = np.load('item_emb_Reg.npy')  # full trained emb
	user_embs = np.load('user_emb_128_Reg.npy')  # full trained emb

	item_emb_weight = torch.FloatTensor(item_embs).cuda()
	# item_embedding = nn.Embedding.from_pretrained(item_emb_weight)
	user_emb_weight = torch.FloatTensor(user_embs).cuda()
	data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
	ITEM_NUM = data_generator.n_items
	USER_NUM = data_generator.n_users
	print('#u', USER_NUM)

	users_to_test = list(data_generator.test_set.keys())
	item_groups = torch.split(item_emb_weight, G, dim=0)

	selection_result = np.load('election_result.npy')
	result_lgcn = test(users_to_test, user_emb_weight, item_emb_weight, max_num_blocks_per, G, selection_result)
	print(result_lgcn)


