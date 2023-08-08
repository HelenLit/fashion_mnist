import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # Check the loss
    if(logs.get('loss') < 0.4):
      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True


print("Version of Tensorflow: " + tf.__version__)

# Instantiate class
myCallback = myCallback()

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

index = 30001

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
#plt.imshow(training_images[index])

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0


# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=myCallback)
#Usually 2 epoch is enough

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])


