
import tensorflow as tf
import numpy as np



(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32,32,3)),
        tf.keras.layers.Conv2D(32, kernel_size= (3,3) , activation = "relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, kernel_size= (3,3) , activation = "relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, kernel_size= (3,3) , activation = "relu"),
        tf.keras.layers.MaxPooling2D(pool_size =(2,2)),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512,activation = "relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128,activation = "relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64,activation = "relu"),
        tf.keras.layers.Dense(32,activation = "relu"),
        tf.keras.layers.Dense(10)

    ])

    model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate =0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy']
        ),

    model.fit(
            x_train,y_train,
            epochs = 10,
            batch_size = 64,
            validation_data=(x_test, y_test)
        )

    model.save("cnn_cifar10.keras")
 
    return model

if __name__ == "__main__":
    model =  build_model()
    model.summary()


 
