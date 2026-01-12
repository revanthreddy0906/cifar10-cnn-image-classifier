
import tensorflow as tf
import numpy as np



(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

def build_model():

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]) 
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32,32,3)),
        tf.keras.layers.Conv2D(32, kernel_size= (3,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, kernel_size= (3,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, kernel_size= (3,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size =(2,2)),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256,activation = "relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128,activation = "relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate =0.0003),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy']
        ),
    
    x_train_aug = data_augmentation(x_train, training = True )

    model.fit(
            x_train_aug,y_train,
            epochs = 12,
            batch_size = 64,
            validation_data=(x_test, y_test)
        ),

    model.export("cnn_cifar10_savedmodel")

    return model

if __name__ == "__main__":
    model =  build_model()
    model.summary()


 
