import numpy as np
from utils.utils import get_data, split_data
import argparse
from pmf.model import PMF
from pmf.constrained_pmf import CPMF
SEED = np.random.randint(0, 1000)
print("Seed is ", SEED)
np.random.seed(SEED)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="CPMF",
                        type=str, help="Algorithm to use")
    parser.add_argument("--mode", default="netflix",
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
    parser.add_argument("--features", default=10,
                        type=int, help="Number of latent features")
    parser.add_argument("--test_ratio", default=0.2,
                        type=float, help="Ratio of size of test dataset")

    args = parser.parse_args()

    # Using default netflix dataset, using url to download data or load data from path.
    if args.mode == "netflix":
        data = get_data()

    elif args.mode == "download":
        data = get_data(args.mode, args.data_url)

    else:
        data = get_data(args.mode, "", args.data_path)

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

    if args.algorithm == "PMF":
        PMF_experiment = PMF(params)
        PMF_experiment.fit(train, test)
        PMF_experiment.plot_loss()

    #Adaptive Priors
    elif args.algorithm == "CPMF":
        CPMF_experiment = CPMF(params)
        CPMF_experiment.fit(train, test)
        CPMF_experiment.plot_loss()

    else:
        print("Invalid algorithm")




