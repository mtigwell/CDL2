import numpy as np
from data import get_item_matrix, get_rating_matrix
from cdl import CDL

K = 50
batch = 256
dropout = 0.1

lamU = 1
lamV = 10
lamW = 10
lv = 0.01

dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for ds in dataset:
    usersfile = '../data/citeulike/citeulike-a/users-{}.dat'.format(ds)
    rating_matrix = '../data/citeulike/citeulike-a/rating_matrix_{}.pickle'.format(ds)
    result_directory = 'results/{}'.format(ds)              

    item_matrix = get_item_matrix()
    rating_matrix = get_rating_matrix(usersfile, rating_matrix)

    cdl = CDL(rating_matrix, item_matrix, lambda_u=lamU, lambda_v=lamV, lambda_w=lamW, lv=lv, K=50, epochs=15, batch=256, dir_save=result_directory, dropout=0.1, recall_m=100,
            trained_matrix=None, pretrain=0
        )       
    cdl.build_model()
    cdl.training(rating_matrix)

