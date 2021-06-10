import numpy as np
import pickle

# for citeulike dataset
def load_data():
    formatted_data = []
    data_list = [i.strip().split(" ") for i in open(r"../data/citeulike/citeulike-a/users.dat").readlines()]

    for i, elem in enumerate(data_list):
        for e in elem[1:]:
            # format: [UserID, MovieID, Rating]
            formatted_data.append([i, int(e), 1])

    return np.array(formatted_data)
