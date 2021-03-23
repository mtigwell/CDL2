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


for lamU in [0.01, 0.1, 1, 10]:
    for lamV in [0.1, 1, 10, 100]:
        for lamW in [0.1, 1, 10]:
            for lv in [0.0001, 0.001, 0.01, 0.1, 1]:
                result_directory = 'results/U{}V{}W{}LV{}'.format(lamU, lamV, lamW, lv)                
                cdl = CDL(rating_matrix, item_matrix, lambda_u=lamU, lambda_v=lamV, lambda_w=lamW, lv=lv, K=50, epochs=15, batch=256, dir_save=result_directory, dropout=0.1, recall_m=100)
                cdl.build_model()
                cdl.training(rating_matrix)
