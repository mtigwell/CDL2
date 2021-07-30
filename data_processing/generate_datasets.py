import numpy as np
import pandas as pd

# remove the top 10% most dense nodes
def create_dataset(usersfile, newfilename, removal_rate):
    '''
    usersfile: input file to pull from
    newfilename: new users file to write to
    removal rate: the percentage of users to keep. EX: a value of 0.9 would retain the 90% of the dataset that is least dense
    '''

    user_conn = []
    datafile = ''
    # usersfile = '../data/citeulike/citeulike-a/users.dat'

    with open(usersfile) as rating_file:
        print('Old dataset has {} users'.format(len(rating_file.readlines())))

    # get connection threshold
    with open(usersfile) as rating_file:
        lines = rating_file.readlines()
        for _, line in enumerate(lines):
            items = line.strip().split(" ")
            user_conn.append(len(items))

    cutoff = int(len(user_conn) * removal_rate)
    user_conn.sort()
    thres = user_conn[cutoff] # the threshold for max number of connections for a user to retain in the dataset
    print('thres', thres)

    # remove dense users
    with open(usersfile) as rating_file:
        lines = rating_file.readlines()
        for _, line in enumerate(lines):
            items = line.strip().split(" ")
            if len(items) > thres:
                continue
            else:
                datafile += line

    # write to new data file
    newfile = open(newfilename, 'w')
    newfile.write(datafile)
    newfile.close()
    
    with open(newfilename) as nf:
        print('New dataset has {} users'.format(len(nf.readlines())))
    
    calculate_sparsity(newfilename)


def calculate_sparsity(users_file):
    count = 0
    items_file = "../data/citeulike/citeulike-a/mult.dat"
    
    with open(items_file) as item_info_file:
        item_size = len(item_info_file.readlines())

    with open(users_file) as rating_file:
        lines = rating_file.readlines()
        users = len(lines)
        
        for _, line in enumerate(lines):
            items = line.strip().split(" ")
            count += len(items)

    sparsity = (1 - (count / (item_size * users))) * 100

    # print(item_size)
    # print(count)
    # print(users)
    print("Sparsity: {}%".format(sparsity))

calculate_sparsity('../data/citeulike/citeulike-a/users.dat')
create_dataset('../data/citeulike/citeulike-a/users.dat', '../data/citeulike/citeulike-a/users-1.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-1.dat', '../data/citeulike/citeulike-a/users-2.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-2.dat', '../data/citeulike/citeulike-a/users-3.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-3.dat', '../data/citeulike/citeulike-a/users-4.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-4.dat', '../data/citeulike/citeulike-a/users-5.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-5.dat', '../data/citeulike/citeulike-a/users-6.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-6.dat', '../data/citeulike/citeulike-a/users-7.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-7.dat', '../data/citeulike/citeulike-a/users-8.dat', 0.9)
create_dataset('../data/citeulike/citeulike-a/users-8.dat', '../data/citeulike/citeulike-a/users-9.dat', 0.9)

