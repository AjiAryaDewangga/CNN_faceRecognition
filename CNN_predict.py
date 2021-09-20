import tensorflow as tf


def Model():
    IMG_SHAPE=(200, 200, 3)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=7, activation='relu', input_shape=IMG_SHAPE ))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=54, kernel_size=5, activation='relu', input_shape= IMG_SHAPE))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
    model.load_weights('data/Weight/Weight_FR.h5')

    return model





        


