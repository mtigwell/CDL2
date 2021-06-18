import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim, encoder_activation, decoder_activation):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation

        self.encoder = tf.keras.Sequential([
            layers.Dense(latent_dim, activation=self.encoder_activation),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(input_dim, activation=self.decoder_activation),
        ])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SDAE(Model):
    def __init__(self, ae_layers):
        super(SDAE, self).__init__()
        self.ae_layers = ae_layers
        self.models = []

    def make(self):
        for i, layer in enumerate(self.ae_layers[:-1]):
            print('building layer input {} output {}'.format(layer, self.ae_layers[i+1]))
            m = Autoencoder(layer, self.ae_layers[i+1], 'relu', 'sigmoid')
            # m.compile(optimizer='adam', loss=losses.MeanSquaredError())
            self.models.append(Autoencoder(layer, self.ae_layers[i+1], 'relu', 'sigmoid'))

    def call(self, train, test, epochs):
        train_set = train
        test_set = test

        for m in self.models:
            m.compile(optimizer='adam', loss=losses.MeanSquaredError())
            noise = np.random.normal(0, .1, train_set.shape)
            x_train_corrupt = train_set + noise
            m.fit(x_train_corrupt, train_set, epochs=epochs, shuffle=True, validation_data=(test_set, test_set))
            train_set = m.encode(train_set)
            test_set = m.encode(test_set)

    def get_layers(self):
        model_layers = []
        for m in self.models:
            w = m.get_weights()
            layer_dict = {
                'w1': w[0],
                'b1': w[1],
                'w2': w[2],
                'b2': w[3]
            }
            model_layers.append(layer_dict)
        return model_layers

