import tensorflow as tf
import numpy as np
from Model_CNN import build_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.models.load_model("cnn_cifar10.keras")
x_test = x_test.astype("float32") / 255.0
logits = model.predict(x_test)
y_pred = np.argmax(logits, axis=1)
y_test = y_test.flatten()

cm = confusion_matrix(y_test, y_pred)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot()
plt.show()

