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


# from SDAE import SDAE
# 
# SPLIT = 0.8 #80/20
# split = int(item_matrix.shape[0] * SPLIT)
# x_train = item_matrix[:split]
# x_test = item_matrix[split:]

# ae_layers = [item_matrix.shape[1], 200, 50]

# sdae = SDAE(ae_layers)
# sdae.make()
# sdae.call(x_train, x_test, epochs=10)
# trained_model = sdae.get_layers()


from cdl import CDL

K = 50
batch = 256
dropout = 0.1


# lambda hyperparameter search
for lamU in [0.1, 1, 10]:
    for lamV in [1, 10, 100]:
        for lamW in [0.1, 1, 10]:
            for lv in [0.001, 0.01, 0.1]:
                result_directory = 'results/cdl/pretrained/U{}V{}W{}LV{}'.format(lamU, lamV, lamW, lv)                
                cdl = CDL(rating_matrix, item_matrix, lambda_u=lamU, lambda_v=lamV, lambda_w=lamW, lv=lv, K=50, epochs=15, batch=256, dir_save=result_directory, dropout=0.1, recall_m=100,
                        trained_matrix=None, pretrain=0
                    )       
                cdl.build_model()
                cdl.training(rating_matrix)

