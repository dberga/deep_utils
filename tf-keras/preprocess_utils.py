import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pdb

#default decode/preprocess params
RZ_IMG_HEIGHT=224
RZ_IMG_WIDTH=224


@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
  # convert the compressed string to a 3D uint8 tensor
  example = tf.image.decode_jpeg(example, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  example = tf.image.convert_image_dtype(example, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(example, [RZ_IMG_HEIGHT, RZ_IMG_WIDTH])

@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image_bw(example, feature):
  # convert the compressed string to a 3D uint8 tensor
  example = tf.image.decode_jpeg(example, channels=3)
  #convert to gray and copy to 3 channels
  example = tf.image.rgb_to_grayscale(example)
  example = tf.image.grayscale_to_rgb(example)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  example = tf.image.convert_image_dtype(example, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(example, [RZ_IMG_HEIGHT, RZ_IMG_WIDTH])


#mapping functions
def rgb2gray(example):
    image = example['image']
    image = tf.image.rgb_to_grayscale(image, )
    #image = tf.image.convert_image_dtype(image, tf.float32)
    return image

