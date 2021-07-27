import numpy as np
import matplotlib.pyplot as plt

class PMF:
    def __init__(self, params):
        self.params = params
        self.n_users = params["users"]
        self.lamda = params["lambda"]
        self.n_products = params["products"]
        self.mean_rat = 0
        self.train_res = []
        self.test_res = []

    def fit(self, train_data, test_data, rating_matrix):
        # print('min: {}, max: {}'.format(np.min(train_data[:, 1]), np.max(train_data[:, 1])))

        epochs = self.params["epoch"]
        lr = self.params["lr"]
        lamda = self.params["lambda"]
        n_features = self.params["features"]
        batch_size = self.params["batch_size"]
        momentum = self.params["momentum"]
        w_users = 0.1*np.random.randn(self.n_users, n_features)
        w_products = 0.1*np.random.randn(self.n_products, n_features)
        self.mean_rat = np.mean(train_data[:, 2])
        update_u = np.zeros((self.n_users, n_features))
        update_p = np.zeros((self.n_products, n_features))

        print("starting")
        for epoch in range(epochs):
            print("Epoch: ", epoch, end=" ")
            np.random.shuffle(train_data)

            for batch in range(round(train_data.shape[0]/batch_size)):
                train_batch = train_data[batch*batch_size:(batch+1)*batch_size]
                users = train_batch[:, 0].astype(np.int)
                products = train_batch[:, 1].astype(np.int)
                ratings = train_batch[:, 2] - self.mean_rat
                pred = np.sum(np.multiply(w_users[users, :], w_products[products, :]), 1)
                l1_error = pred-ratings
                batch_grad_u = 2*np.multiply(l1_error[:, np.newaxis], w_products[products])+lamda*w_users[users]
                batch_grad_p = 2*np.multiply(l1_error[:, np.newaxis], w_users[users]) + lamda*w_products[products]
                dw_u = np.zeros((self.n_users, n_features))
                dw_p = np.zeros((self.n_products, n_features))
                for r in range(batch_size):
                    try:
                        dw_u[users[r]] += batch_grad_u[r]
                    except:
                        continue
                    try:
                        dw_p[products[r]] += batch_grad_p[r]
                    except:
                        continue
                update_u = momentum*update_u + lr*dw_u/batch_size
                update_p = momentum*update_p + lr*dw_p/batch_size
                w_users -= update_u
                w_products -= update_p

            u_idx = train_data[:, 0].astype(np.int)
            p_idx = train_data[:, 1].astype(np.int)
            training_loss = self.compute_rmse(w_users[u_idx], w_products[p_idx], train_data[:, 2])
            print("finished. Current training RMSE: ", training_loss, end=" ")
            self.train_res.append(training_loss)
            u_idx = test_data[:, 0].astype(np.int)
            p_idx = test_data[:, 1].astype(np.int)
            test_loss = self.compute_rmse(w_users[u_idx], w_products[p_idx], test_data[:, 2])
            print("Current validation RMSE: ", test_loss)
            self.test_res.append(test_loss)

            prediction_matrix = np.dot(w_users, w_products.T)
            print('Validation Recall: {}'.format(self.get_recall(rating_matrix.T, prediction_matrix.T, 100)))

    def compute_obj(self, w_u, w_p, rat):
        w_u = w_u
        w_p = w_p
        pred = np.sum(np.multiply(w_u, w_p), 1) + self.mean_rat
        error = np.linalg.norm(pred - rat)**2 + \
                0.5 * self.lamda*(np.linalg.norm(w_u)**2 + np.linalg.norm(w_p)**2)
        return error/rat.shape[0]

    def compute_rmse(self, w_u, w_p, rat):
        pred = np.sum(np.multiply(w_u, w_p), 1) + self.mean_rat
        err = rat - pred
        loss = np.linalg.norm(err)/np.sqrt(pred.shape[0]) 
        return loss

    def get_recall(self, ratings, pred, m):
        ''' 
        inputs: 
            ratings matrix: [users, items]
            pred: floats [users, items]
            m: recall@M
        '''
        # generate number of items user likes among top M
        b = list()
        top_m = np.argpartition(pred, -m)[:, -m:]

        for i, row in enumerate(ratings):
            newrow = np.take(row, top_m[i])
            b.append(newrow)

        b = np.array(b)
        # b = b.squeeze(axis=1)
        good_picks = np.sum(np.array(b), axis=1)

        # total number user likes
        user_picks = ratings.sum(axis=1)

        # generate recall
        user_recall = np.divide(good_picks, user_picks, where=user_picks != 0)
        recall = np.mean(user_recall)
        return recall


    # def get_recall(self, w_u, w_p, ratings, m):
    #     ''' 
    #     inputs: 
    #         ratings matrix: [users, items]
    #         pred: floats [users, items]
    #         m: recall@M
    #     '''
    #     pred = np.sum(np.multiply(w_u, w_p), 1) + self.mean_rat

    #     # generate number of items user likes among top M
    #     b = list()
    #     top_m = np.argpartition(pred, -m)[:, -m:]

    #     for i, row in enumerate(ratings):
    #         newrow = np.take(row, top_m[i])
    #         b.append(newrow)

    #     b = np.array(b)
    #     b = b.squeeze(axis=1)
    #     good_picks = np.sum(np.array(b), axis=1)

    #     # total number user likes
    #     user_picks = ratings.sum(axis=1)

    #     # generate recall
    #     user_recall = np.divide(good_picks, user_picks, where=user_picks != 0)
    #     recall = np.mean(user_recall)
    #     return recall


    def plot_loss(self):
        plt.plot(np.arange(self.params["epoch"]), self.train_res, color="red", label="Training Loss")
        plt.plot(np.arange(self.params["epoch"]), self.test_res, color="blue", label="Test Loss")
        plt.legend()
        plt.grid()
        plt.title("RMSE Loss vs Epoch for Vanilla PMF")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


