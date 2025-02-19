import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
modelT= tf.keras.models.load_model("MNIST_Transformed.keras")
model= tf.keras.models.load_model("MNIST.keras")

ysT = modelT.evaluate(x_test, y_test, verbose=2)
ys = model.evaluate(x_test, y_test, verbose=2)

plt.violinplot(ys)
plt.violinplot(ysT)
plt.savefig("final.png")
