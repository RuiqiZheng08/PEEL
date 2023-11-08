K = 5000 # num of combinations
max_num_blocks_per = 16 # the max num of blocks per item
G = 20 # num of groups, 0.146MB per block
M = 34 # memory upperbound (converted to the max num of total blocks) 34, 68, 171

# for evaluator_rule.py
BATCH_SIZE = 256
# ITEM_NUM = 91381


# the same param names as they are in the paper
P = 20
C = 50
S = 5


# for nfm

# dataset name
dataset = 'yelp2020'
# assert dataset in ['ml-tag', 'frappe']

# model name
model = 'NFM'
assert model in ['FM', 'NFM']

# as the log_loss is not implemented in Xiangnan's repo, I thus remove this loss type
loss_type = 'square_loss'
assert loss_type in ['square_loss']

# important settings (normally default is the paper choice)
optimizer = 'Adagrad'
activation_function = 'relu'
assert optimizer in ['Adagrad', 'Adam', 'SGD', 'Momentum']
assert activation_function in ['relu', 'sigmoid', 'tanh', 'identity']

# paths
main_path = 'nfm/data/{}/'.format(dataset)
# if dataset == 'try':
train_libfm = main_path + 'train.txt'
valid_libfm = main_path + 'valid.txt'
test_libfm = main_path + 'test.txt'
# else:
#     train_libfm = main_path + '{}.train.libfm'.format(dataset)
#     valid_libfm = main_path + '{}.validation.libfm'.format(dataset)
#     test_libfm = main_path + '{}.test.libfm'.format(dataset)



model_path = 'nfm/models/'
# FM_model_path = model_path + 'NFM.pth' # useful when pre-train
NFM_model_path = model_path + 'NFM'+dataset+'.pth'
NFM_model = 'NFM.pth'
Ks = [20, 40, 50, 80, 100]