import tensorflow as tf
import tensorflow_datasets as tfds


split = (80, 10, 10)
(cat_dog_train, cat_dog_valid, cat_dog_test), info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[10%:]', 'train[10%:]'], with_info=True, as_supervised=True)

IMAGE_SIZE = 100

def pre_process_image(image, label):
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
  return image, label

TRAIN_BATCH_SIZE = 64
cat_dog_train = cat_dog_train.map(pre_process_image).shuffle(1000).repeat().batch(TRAIN_BATCH_SIZE)
cat_valid = cat_dog_train.map(pre_process_image).repeat().batch(1000)

from transfer_model import build_resnet_model

transfer_learning_model = build_resnet_model()
