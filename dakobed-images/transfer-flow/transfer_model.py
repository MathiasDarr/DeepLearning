import tensorflow as tf
from tensorflow.keras import layers

def build_standard_model():
    IMAGE_SIZE = 100

    head = tf.keras.Sequential()
    head.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(32, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(64, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))


    average_pool = tf.keras.Sequential()
    average_pool.add(layers.AveragePooling2D())
    average_pool.add(layers.Flatten())
    average_pool.add(layers.Dense(1, activation='sigmoid'))

    standard_model = tf.keras.Sequential([
        head,
        average_pool
    ])

    standard_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return standard_model

def build_resnet_model():
    IMAGE_SIZE = 100
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    res_net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    global_average_layer = layers.GlobalAveragePooling2D()
    output_layer = layers.Dense(1, activation='sigmoid')
    transfer_learning_model = tf.keras.Sequential([
        res_net,
        global_average_layer,
        output_layer
    ])
    return transfer_learning_model
