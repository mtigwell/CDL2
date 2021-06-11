import numpy as np
import pickle
from cdl import CDL
import data

try:
    print('Loading item data...')
    with open(r'../data/citeulike/citeulike-a/item_bow.pickle', 'rb') as handle:
        item_matrix = pickle.load(handle) 
    print('Loading rating matrix...')
    with open(r'../data/citeulike/citeulike-a/rating_matrix.pickle', 'rb') as handle2:
        rating_matrix = pickle.load(handle2)
except:
    print('preprocessing data...')
    data.preprocess_data()
    with open(r'../data/citeulike/citeulike-a/item_bow.pickle', 'rb') as handle:
        item_matrix = pickle.load(handle) 
    with open(r'../data/citeulike/citeulike-a/rating_matrix.pickle', 'rb') as handle2:
        rating_matrix = pickle.load(handle2)


# lambda hyperparameter search
# for lamU in [0.1, 1, 10]:
#     for lamV in [1, 10, 100]:
#         for lamW in [0.1, 1, 10]:
#             for lv in [0.001, 0.01, 0.1]:
#                 result_directory = 'results/U{}V{}W{}LV{}'.format(lamU, lamV, lamW, lv)                
#                 cdl = CDL(rating_matrix, item_matrix, lambda_u=lamU, lambda_v=lamV, lambda_w=lamW, lv=lv, K=50, epochs=15, batch=256, dir_save=result_directory, dropout=0.1, recall_m=100)
#                 cdl.build_model()
#                 cdl.training(rating_matrix)


# for dropout in [0.05, 0.1, 0.2]:
#     for K in [50]:
#         for batch in [32, 256, 1024]:
#             result_directory = 'results/drop{}K{}batch{}'.format(dropout, K, batch)                
#             cdl = CDL(rating_matrix, item_matrix, lambda_u=1, lambda_v=10, lambda_w=10, lv=0.01, K=K, epochs=15, batch=batch, dir_save=result_directory, dropout=dropout, recall_m=100)
#             cdl.build_model()
#             cdl.training(rating_matrix)

K = 50
batch = 256
dropout = 0.1


from SDAE import SDAE

SPLIT = 0.8 #80/20
split = int(item_matrix.shape[0] * SPLIT)
x_train = item_matrix[:split]
x_test = item_matrix[split:]

ae_layers = [784, 64, 16]
sdae = SDAE(ae_layers)
sdae.make()
sdae.call(x_train, x_test, epochs=2)
a = sdae.get_layers()

result_directory = 'results/testrun'
cdl = CDL(rating_matrix, item_matrix, lambda_u=1, lambda_v=10, lambda_w=10, lv=0.01, K=K, epochs=15, batch=batch, dir_save=result_directory, dropout=dropout, recall_m=100)
cdl.build_model()
# cdl.pretrain()
cdl.training(rating_matrix)
