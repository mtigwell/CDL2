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


#build rating_matrix
def generate_rating_matrix(users, items):
    rating_matrix = np.zeros((users, items))
    print('Making rating matrix..')

    with open(r"../data/citeulike/citeulike-a/users.dat") as rating_file:
        lines = rating_file.readlines()
        for index,line in enumerate(lines):
            items = line.strip().split(" ")
            for item in items:
                rating_matrix[index][int(item)] = 1
    return rating_matrix
