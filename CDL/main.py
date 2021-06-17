import numpy as np
import pickle
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



from SDAE import SDAE

SPLIT = 0.8 #80/20
split = int(item_matrix.shape[0] * SPLIT)
x_train = item_matrix[:split]
x_test = item_matrix[split:]

ae_layers = [item_matrix.shape[1], 200, 50]

sdae = SDAE(ae_layers)
sdae.make()
sdae.call(x_train, x_test, epochs=10)
trained_model = sdae.get_layers()

# pretrain = 0

from cdl import CDL

K = 50
batch = 256
dropout = 0.1


result_directory = 'results/test_raw'
cdl = CDL(rating_matrix, item_matrix, lambda_u=1, lambda_v=10, lambda_w=10, lv=0.01, K=K, epochs=15, batch=batch, 
        dir_save=result_directory, dropout=dropout, recall_m=100, trained_matrix=None, pretrain=0
    )
cdl.build_model()
cdl.training(rating_matrix)


result_directory = 'results/test_pretrain'
cdl = CDL(rating_matrix, item_matrix, lambda_u=1, lambda_v=10, lambda_w=10, lv=0.01, K=K, epochs=15, batch=batch, 
        dir_save=result_directory, dropout=dropout, recall_m=100, trained_matrix=trained_model, pretrain=1
    )
cdl.build_model()
cdl.training(rating_matrix)