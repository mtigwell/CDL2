import tensorflow as tf
import os
import csv
import numpy as np
import pickle
from mf import MF
from utils import add_noise
from datetime import datetime

np.random.seed(5)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_eager_execution()


# MODEL
class CDL():
    '''
    Parameters:
        lambda_u: regularization coefficent for user latent matrix U
        lambda_v: regularization coefficent for item latent matrix V
        lv: lambda_v/lambda_n in CDL; this controls the trade-off between reconstruction error in pSDAE and recommendation accuracy during training
        K: number of latent factors
        num_iter: number of iterations (minibatches) to train (a epoch in the used dataset takes about 68 iterations)
        batch_size: minibatch size
        dir_save: directory to save training results
    '''

    def __init__(
        self,
            rating_matrix, item_infomation_matrix,
            lambda_u, lambda_v, lambda_w, lv, K, epochs, batch, dir_save, dropout, recall_m
    ):

        self.n_input = 8000
        self.n_hidden1 = 200
        self.n_hidden2 = 50
        self.k = K

        self.lv = lv
        self.lambda_w = lambda_w
        self.lambda_n = lambda_v / lv
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

        self.drop_ratio = dropout
        self.learning_rate = 0.001
        self.m = recall_m
        self.epochs = epochs
        self.batch_size = batch
        self.dir_save = dir_save

        self.num_u = rating_matrix.shape[0]
        self.num_v = rating_matrix.shape[1]

        self.Weights = {
            'w1': tf.Variable(tf.random.normal([self.n_input, self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'w2': tf.Variable(tf.random.normal([self.n_hidden1, self.n_hidden2], mean=0.0, stddev=1 / self.lambda_w)),
            'w3': tf.Variable(tf.random.normal([self.n_hidden2, self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'w4': tf.Variable(tf.random.normal([self.n_hidden1, self.n_input], mean=0.0, stddev=1 / self.lambda_w))
        }
        self.Biases = {
            'b1': tf.Variable(tf.random.normal([self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'b2': tf.Variable(tf.random.normal([self.n_hidden2], mean=0.0, stddev=1 / self.lambda_w)),
            'b3': tf.Variable(tf.random.normal([self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'b4': tf.Variable(tf.random.normal([self.n_input], mean=0.0, stddev=1 / self.lambda_w))
        }
        
        self.make_directory()
        self.item_infomation_matrix = item_infomation_matrix
        self.build_model()

    def encoder(self, x, drop_ratio):
        w1 = self.Weights['w1']
        b1 = self.Biases['b1']
        L1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        L1 = tf.nn.dropout(L1, rate=1 - (1 - drop_ratio))

        w2 = self.Weights['w2']
        b2 = self.Biases['b2']
        L2 = tf.nn.sigmoid(tf.matmul(L1, w2) + b2)
        L2 = tf.nn.dropout(L2, rate=1 - (1 - drop_ratio))

        return L2

    def decoder(self, x, drop_ratio):
        w3 = self.Weights['w3']
        b3 = self.Biases['b3']
        L3 = tf.nn.sigmoid(tf.matmul(x, w3) + b3)
        L3 = tf.nn.dropout(L3, rate=1 - (1 - drop_ratio))

        w4 = self.Weights['w4']
        b4 = self.Biases['b4']
        L4 = tf.nn.sigmoid(tf.matmul(L3, w4) + b4)
        L4 = tf.nn.dropout(L4, rate=1 - (1 - drop_ratio))

        return L4

    def build_model(self):
        self.model_X_0 = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.n_input))
        self.model_X_c = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.n_input))
        self.model_V = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.k))
        self.model_drop_ratio = tf.compat.v1.placeholder(tf.float32)

        self.V_sdae = self.encoder(self.model_X_0, self.model_drop_ratio)
        self.y_pred = self.decoder(self.V_sdae, self.model_drop_ratio)

        self.Regularization = tf.reduce_sum(input_tensor=[tf.nn.l2_loss(
            w)+tf.nn.l2_loss(b) for w, b in zip(self.Weights.values(), self.Biases.values())])
        loss_r = 1/2 * self.lambda_w * self.Regularization
        loss_a = 1/2 * self.lambda_n * \
            tf.reduce_sum(input_tensor=tf.pow(self.model_X_c - self.y_pred, 2))
        loss_v = 1/2 * self.lambda_v * \
            tf.reduce_sum(input_tensor=tf.pow(self.model_V - self.V_sdae, 2))
        self.Loss = loss_r + loss_a + loss_v

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(self.Loss)

    def training(self, rating_matrix):
        # np.random.shuffle(self.item_infomation_matrix) #random index of train data
        self.item_infomation_matrix_noise = add_noise(
            self.item_infomation_matrix, 0.3)

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        mf = MF(rating_matrix)

        for epoch in range(self.epochs):
            start = datetime.now()
            print("%d / %d" % (epoch+1, self.epochs))

            V_sdae = sess.run(self.V_sdae, feed_dict={
                              self.model_X_0: self.item_infomation_matrix_noise, self.model_drop_ratio: self.drop_ratio})

            U, V = mf.ALS(V_sdae)
            V = np.resize(V, (16980, 50))

            for i in range(0, self.item_infomation_matrix.shape[0], self.batch_size):
                X_train_batch = self.item_infomation_matrix_noise[i:i+self.batch_size]
                y_train_batch = self.item_infomation_matrix[i:i +
                                                            self.batch_size]
                V_batch = V[i:i+self.batch_size]
                _, my_loss = sess.run([self.optimizer, self.Loss], feed_dict={
                                      self.model_X_0: X_train_batch, self.model_X_c: y_train_batch, self.model_V: V_batch, self.model_drop_ratio: self.drop_ratio})

            # get recall
            R = U.T * V.T  # predictions
            recall = self.get_recall(rating_matrix.T, R.T, self.m)

            print('Loss: {}'.format(my_loss))
            print('Recall: {}'.format(recall))
            print('Epoch time: {}'.format(datetime.now() - start))

            self.write_results(epoch, my_loss, recall, datetime.now() - start)

            # print('V_SDAE {}'.format(V_sdae.shape))
            # print('V shape {}'.format(V.shape))
            # print('U Shape {}'.format(U.shape))
            # print('R shape {}'.format(R.shape))
            # print('Ratings Matrix shape {}'.format(rating_matrix.shape))

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
            # print('Shape row {}'.format(row.shape))
            # print('Shape top_m {}'.format(top_m[i].shape))
            # print('Shape new row {}'.format(newrow.shape))
            b.append(newrow)

        b = np.array(b)
        b = b.squeeze(axis=1)
        good_picks = np.sum(np.array(b), axis=1)

        # print('Shape b {}'.format(b.shape))
        # print('Shape good picks {}'.format(good_picks.shape))

        # total number user likes
        user_picks = ratings.sum(axis=1)

        # generate recall
        user_recall = np.divide(good_picks, user_picks, where=user_picks != 0)
        recall = np.mean(user_recall)
        return recall

    def write_results(self, ep, loss, recall, time):
        with open('{}/errors.csv'.format(self.dir_save), mode='a') as f:
            fieldnames = ['Epoch', 'Loss', 'Recall', 'Time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # writer.writeheader()
            writer.writerow({'Epoch': ep, 'Loss': loss,
                            'Recall': recall, 'Time': time})

    # create directory
    def make_directory(self):
        if not os.path.isdir(self.dir_save):
            os.system('mkdir %s' % self.dir_save)
        fp = open(self.dir_save+'/cdl.log', 'w')
        print('lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (self.lambda_v, self.lambda_u, self.lv, self.k))
        fp.write('lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' %
                (self.lambda_v, self.lambda_u, self.lv, self.k))
        fp.close()

        with open('{}/errors.csv'.format(self.dir_save), mode='a') as f:
            fieldnames = ['Epoch', 'Loss', 'Recall', 'Time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # def testing(self, rating_matrix):
    #     sess = tf.compat.v1.Session()
    #     sess.run(tf.compat.v1.global_variables_initializer())

    #     self.test_rating_noise = add_noise(self.item_infomation_matrix , 0.3)

    #     rating_matrix # true label

    #     _ , my_loss = sess.run([self.optimizer, self.Loss, self.recall] , feed_dict={self.model_X_0 :X_train_batch , self.model_X_c : y_train_batch , self.model_V:V_batch, self.model_drop_ratio : 0.1})
