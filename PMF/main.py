import numpy as np
from numpy.core.fromnumeric import prod
from utils.utils import get_data, split_data
from utils.dataloader import load_data, generate_rating_matrix
import argparse
from pmf.model import PMF
from pmf.constrained_pmf import CPMF
SEED = np.random.randint(0, 1000)
print("Seed is ", SEED)
np.random.seed(SEED)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="PMF",
                        type=str, help="Algorithm to use")
    parser.add_argument("--mode", default="citeulike",
                        type=str, help="Download data mode or load data mode")
    parser.add_argument("--data_url",
                        type=str, help="Url for rating data")
    parser.add_argument("--data_path",
                        type=str, help="Folder for existing data")
    parser.add_argument("--epoch", default=20,
                        type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=1000,
                        type=int, help="Size of batches")
    parser.add_argument("--lamda", default=.01,
                        type=float, help="Regularization parameter")
    parser.add_argument("--momentum", default=.8,
                        type=float, help="Momentum for SGD")
    parser.add_argument("--lr", default=10,
                        type=float, help="Learning rate parameter")
    parser.add_argument("--features", default=50,
                        type=int, help="Number of latent features")
    parser.add_argument("--test_ratio", default=0.2,
                        type=float, help="Ratio of size of test dataset")

    args = parser.parse_args()

    if args.mode == "citeulike":
        data = load_data()
        users = np.unique(data[:, 0]).shape[0]
        products = np.unique(data[:, 1]).shape[0]
        rating_mat = generate_rating_matrix(users, products)

    train, test = split_data(data, SEED, args.test_ratio)

    params = dict()
    params["epoch"] = args.epoch
    params["lambda"] = args.lamda
    params["momentum"] = args.momentum
    params["batch_size"] = args.batch_size
    params["lr"] = args.lr
    params["features"] = args.features
    params["users"] = np.unique(data[:, 0]).shape[0]
    params["products"] = np.unique(data[:, 1]).shape[0]

    print(params)

    if args.algorithm == "PMF":
        PMF_experiment = PMF(params)
        PMF_experiment.fit(train, test, rating_mat)
        PMF_experiment.plot_loss()
    #Adaptive Priors
    # elif args.algorithm == "CPMF":
    #     CPMF_experiment = CPMF(params)
    #     CPMF_experiment.fit(train, test)
    #     CPMF_experiment.plot_loss()
    else:
        print("Invalid algorithm")
