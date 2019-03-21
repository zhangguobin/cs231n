import tensorflow as tf
from tensorflow.keras import layers

NUM_CLASSES = 1000

class VGG16(object):
    def extract_features(self):
        all_layers = []
        x = layers.Input(shape=(None, None, 3), tensor=self.image)
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
        - input: optional input to the model. If None, will use placeholder for input.
        """
        self.image = tf.placeholder('float',shape=[None,None,None,3],name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')
        self.all_layers = self.extract_features()
        self.features = self.all_layers[-1]
        
        # Classification block
        # x = layers.Flatten(name='flatten')(self.features)
#         x = tf.reshape(self.features,[-1, 4096]) # hack
#         x = layers.Dense(4096, activation='relu', name='fc1')(x)
#         self.all_layers.append(x)
#         x = layers.Dense(4096, activation='relu', name='fc2')(x)
#         self.all_layers.append(x)
#         x = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
#         self.all_layers.append(x)
        
        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)