import tensorflow as tf
from tensorflow.keras import layers

# copied from Keras vgg16.py with some modifications
class VGG16(object):
    def extract_features(self, inputs=None):
        all_layers = []
        if inputs is None:
            inputs = self.image
        x = layers.Input(shape=(None, None, 3), tensor=inputs)
        # Block 1
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv1')(x)
        all_layers.append(x)
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv2')(x)
        all_layers.append(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        all_layers.append(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')(x)
        all_layers.append(x)
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')(x)
        all_layers.append(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        all_layers.append(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1')(x)
        all_layers.append(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2')(x)
        all_layers.append(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3')(x)
        all_layers.append(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        all_layers.append(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1')(x)
        all_layers.append(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(x)
        all_layers.append(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(x)
        all_layers.append(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        all_layers.append(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1')(x)
        all_layers.append(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2')(x)
        all_layers.append(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')(x)
        all_layers.append(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        all_layers.append(x)
        return all_layers

    def __init__(self, save_path=None, sess=None):
        """Create a VGG16 model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        """
        self.image = tf.placeholder('float',shape=[None,None,None,3],name='input_image')
        self.all_layers = self.extract_features(self.image)
        
        if save_path is not None:
            saver = tf.train.Saver()
            tf.get_variable_scope().reuse_variables()
            saver.restore(sess, save_path)