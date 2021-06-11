import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model



class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim, encoder_activation, decoder_activation):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation

        self.encoder = tf.keras.Sequential([
            # layers.Input(shape=(input_dim,)),
            layers.Dense(latent_dim, activation=self.encoder_activation),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(input_dim, activation=self.decoder_activation),
        # layers.Reshape((28, 28))
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
            self.models.append(Autoencoder(layer, self.ae_layers[i+1], 'relu', 'sigmoid'))

    def call(self, train, test, epochs):
        train_set = train
        test_set = test

        for m in self.models:
            m.compile(optimizer='adam', loss=losses.MeanSquaredError())
            
            print('size train test set {} {}'.format(train_set.shape, test_set.shape))
            print('type train test set {} {}'.format(type(train_set), type(test_set.shape)))

            m.fit(train_set, train_set, epochs=epochs, shuffle=True, validation_data=(test_set, test_set))

            print(m.encode(train_set).op.get_attr('value'))
            train_set = tf.make_ndarray(m.encode(train_set).op.get_attr('value')) 
            test_set = tf.make_ndarray(m.encode(test_set).op.get_attr('value'))

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



# # declare variables
# (x_train, _), (x_test, _) = fashion_mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape(60000, 784)
# x_test= x_test.reshape(10000, 784)


# ae_layers = [784, 64, 16]
# sdae = SDAE(ae_layers)
# sdae.make()
# sdae.call(x_train, x_test, epochs=2)
# a = sdae.get_layers()

# print(a[0]['w1'].shape)
