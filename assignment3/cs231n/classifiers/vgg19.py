import tensorflow as tf
from tensorflow import layers

# copied from Keras vgg19.py with some modifications
# ensure input is aligned with default data_format of tensorflow
class VGG19(object):
    def extract_features(self, inputs=None, reuse=tf.AUTO_REUSE):
        all_layers = []
        if inputs is None:
            inputs = self.image
        x = inputs
        # Block 1
        x = layers.conv2d(x, 64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv1',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv2',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block1_pool')
        all_layers.append(x)

        # Block 2
        x = layers.conv2d(x, 128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block2_pool')
        all_layers.append(x)

        # Block 3
        x = layers.conv2d(x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv4',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block3_pool')
        all_layers.append(x)

        # Block 4
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv4',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block4_pool')
        all_layers.append(x)

        # Block 5
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.conv2d(x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv4',
                          reuse=reuse)
        all_layers.append(x)
        x = layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block5_pool')
        all_layers.append(x)
        return all_layers

    def __init__(self, save_path=None, sess=None):
        """Create a VGG19 model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        - input: optional input to the model. If None, will use placeholder for input.
        """
        self.image = tf.placeholder('float',shape=[1,None,None,3],name='input_image') # warning on data format
        self.all_layers = self.extract_features(self.image, reuse=tf.AUTO_REUSE)
        
        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)