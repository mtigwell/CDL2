import zipfile, requests, io
import numpy as np
import os

def get_data(mode = "download", 
    data_url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    data_path="./data/ml-100k/u.data"):

    if mode == "download":
        if not (os.path.isfile(data_path)):
            r = requests.get(data_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("./data")

    elif mode != "load":
        print("Unknown mode.")
        exit()

    data = []
    for line in open(data_path, 'r'):
        (userid, movieid, rating, ts) = line.split('\t')
        uid = np.int(userid)
        mid = np.int(movieid)
        rat = np.float(rating)
        data.append([uid, mid, rat])
    data = np.array(data)
    return data


def split_data(data, random_seed=55, test_ratio=0.2):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    cut = round(test_ratio*data.shape[0])
    return data[cut:], data[:cut]
