import numpy as np
import matplotlib.pyplot as plt

# Probabilistic Matrix Factorization with Adaptive Priors


class CPMF:
    def __init__(self, params):
        self.params = params
        self.n_users = params["users"]+1
        self.lamda = params["lambda"]
        self.n_products = params["products"]+1
        self.mean_rat = 0
        self.K = 0
        self.train_res = []
        self.test_res = []

    def transform_ratings(self, ratings, mode="inverse"):
        if mode == "inverse":
            return ratings*(self.K-1)+1
        else:
            return (ratings-1)/(self.K-1)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, train_data, test_data):
        epochs = self.params["epoch"]
        lr = self.params["lr"]
        lamda = self.params["lambda"]
        n_features = self.params["features"]
        batch_size = self.params["batch_size"]
        momentum = self.params["momentum"]
        Y = 0.1*np.random.randn(self.n_users, n_features)
        V = 0.1*np.random.randn(self.n_products, n_features)
        W = 0.1*np.random.randn(self.n_products, n_features)
        self.K = np.max([np.max(train_data[:, 2]), np.max(test_data[:, 2])])
        train_data[:, 2] = self.transform_ratings(train_data[:, 2], mode="pre-process")
        test_data[:, 2] = self.transform_ratings(test_data[:, 2], mode="pre-process")
        self.mean_rat = np.mean(train_data[:, 2])
        update_Y = np.zeros((self.n_users, n_features))
        update_W = np.zeros((self.n_products, n_features))
        update_V = np.zeros((self.n_products, n_features))
        w_filters = dict()
        for i in np.unique(train_data[:, 0]):
            w_filters[i] = train_data[train_data[:, 0] == i][:, 1].astype(int)

        print("starting")
        for epoch in range(epochs):
            print("Epoch: ", epoch+1, end=" ")
            np.random.shuffle(train_data)

            for batch in range(round(train_data.shape[0]/batch_size)):
                train_batch = train_data[batch*batch_size:(batch+1)*batch_size]
                users = train_batch[:, 0].astype(np.int)
                products = train_batch[:, 1].astype(np.int)
                ratings = train_batch[:, 2]
                U = Y[users, :]
                for u in np.unique(users):
                    ws = np.mean(W[w_filters[u]], axis=0)
                    U[u] += ws

                pred = np.sum(np.multiply(U, V[products, :]), 1)
                g_pred = self.sigmoid(pred)
                l1_error = g_pred-ratings
                sigmoid_grad = np.multiply(g_pred[:, np.newaxis], (1-g_pred)[:, np.newaxis])
                pred_grad_Y = np.multiply(l1_error[:, np.newaxis], V[products])
                pred_grad_V = np.multiply(l1_error[:, np.newaxis], U)
                batch_grad_W = 2*np.multiply(sigmoid_grad, pred_grad_Y)
                batch_grad_Y = 2*np.multiply(sigmoid_grad, pred_grad_Y)
                batch_grad_V = 2*np.multiply(sigmoid_grad, pred_grad_V)
                dw_Y = np.zeros((self.n_users, n_features))
                dw_V = np.zeros((self.n_products, n_features))
                dw_W = np.zeros((self.n_products, n_features))

                ## TODO need to vectorize
                for r in range(batch_size):
                    dw_Y[users[r]] += batch_grad_Y[r]
                    dw_V[products[r]] += batch_grad_V[r]
                    k_list = w_filters[users[r]]
                    dw_W[k_list] += (batch_grad_W[r]/k_list.shape[0])

                dw_W += lamda * W
                dw_Y += lamda * Y
                dw_V += lamda * V
                update_Y = momentum*update_Y + lr*dw_Y/batch_size
                update_V = momentum*update_V + lr*dw_V/batch_size
                Y -= update_Y
                V -= update_V

                update_W = momentum*update_W + lr*dw_W/batch_size
                W -= update_W

            u_idx = train_data[:, 0].astype(np.int)
            p_idx = train_data[:, 1].astype(np.int)
            w_V = V[p_idx]
            w_U = Y[u_idx]
            for u in np.unique(u_idx):
                w_filter = train_data[train_data[:, 0] == u][:, 1].astype(int)
                ws = np.mean(W[w_filter], axis=0)
                w_U[u] += ws
            training_loss = self.compute_rmse(w_U, w_V, train_data[:, 2])
            print("finished. Current training RMSE: ", training_loss, end=" ")
            self.train_res.append(training_loss)
            u_idx = test_data[:, 0].astype(np.int)
            p_idx = test_data[:, 1].astype(np.int)
            w_V = V[p_idx]
            w_U = Y[u_idx]
            test_loss = self.compute_rmse(w_U, w_V, test_data[:, 2])
            print("Current validation RMSE: ", test_loss)
            self.test_res.append(test_loss)

    #def compute_obj(self, w_u, w_p, rat):
    #    w_u = w_u
    #    w_p = w_p
    #    pred = np.sum(np.multiply(w_u, w_p), 1)
    #    g_pred = self.sigmoid(pred)
    #    error = np.linalg.norm(g_pred - rat)**2 + \
    #            0.5 * self.lamda*(np.linalg.norm(w_u)**2 + np.linalg.norm(w_p)**2)
    #    return error/rat.shape[0]

    def compute_rmse(self, U, V, rat):
        pred = np.sum(np.multiply(U, V), 1)
        g_pred = self.sigmoid(pred)
        g_pred = self.transform_ratings(g_pred)
        rat = self.transform_ratings(rat)
        err = rat - g_pred
        loss = np.linalg.norm(err)/np.sqrt(pred.shape[0])
        return loss

    def plot_loss(self):
        plt.plot(np.arange(self.params["epoch"]), self.train_res, color="red", label="Training Loss")
        plt.plot(np.arange(self.params["epoch"]), self.test_res, color="blue", label="Test Loss")
        plt.legend()
        plt.grid()
        plt.title("RMSE Loss vs Epoch for Constrained PMF")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


