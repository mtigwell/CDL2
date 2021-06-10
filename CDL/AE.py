import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


# class SDAE():
#     def __init__(self, layers):
#         self.layers = layers
#         self.models = None

#     def build(self):
#         for layer in self.layers:
#             l = Autoencoder()


class Autoencoder(Model):
  def __init__(self, input_dim, latent_dim, encoder_activation, decoder_activation):
    super(Autoencoder, self).__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.encoder_activation = encoder_activation
    self.decoder_activation = decoder_activation

    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation=self.encoder_activation),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation=self.decoder_activation),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


# declare variables
latent_dim = 64


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

autoencoder = Autoencoder(input_dim, latent_dim, 'relu', 'sigmoid')
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))