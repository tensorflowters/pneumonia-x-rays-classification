import tensorflow as tf

class ConcatenationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConcatenationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.layers.Concatenate()(inputs)