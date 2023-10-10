from functools import partial

import tensorflow as tf
from keras import layers
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE
DefaultConv2D = partial(
    layers.Conv2D,
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal",
)


class GeneralizedSwish(keras.layers.Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = tf.Variable(
            initial_value=beta,
            trainable=kwargs.get("trainable", True),
            dtype=tf.float32,
            name=self.name + "/beta",
        )

    def call(self, inputs):
        return inputs / (1.0 + tf.exp(-self.beta * inputs))

    def get_config(self):
        return super().get_config() | {
            "beta": self.beta.numpy(),
        }


class MaxDepthPool2D(keras.layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)  # The number of channels is stored in shape[-1].
        groups = shape[-1] // self.pool_size  # Number of channel groups.
        new_shape = tf.concat([shape[:-1], [groups, self.pool_size]], axis=0)
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)

    def get_config(self):
        return super().get_config() | {
            "pool_size": self.pool_size,
        }


class SEUnit(keras.layers.Layer):
    def __init__(self, squeeze_factor=16, hidden_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.squeeze_factor = squeeze_factor
        self.hidden_activation = keras.activations.get(hidden_activation)

    def build(self, input_shape):
        n_units_hidden = input_shape[-1] // self.squeeze_factor
        n_units_out = input_shape[-1]
        self.feed_forward = [
            # keepdims=True, because it must be broadcastable for multiplication.
            layers.GlobalAvgPool2D(keepdims=True),
            layers.Dense(n_units_hidden, self.hidden_activation, kernel_initializer="he_normal"),
            layers.Dense(n_units_out, "sigmoid"),
        ]
        super().build(input_shape)

    def call(self, inputs):
        X = inputs
        for layer in self.feed_forward:
            X = layer(X)
        return X * inputs  # Calibrate feature maps.

    def get_config(self):
        return super().get_config() | {
            "squeeze_factor": self.squeeze_factor,
            "hidden_activation": keras.activations.serialize(self.hidden_activation),
        }


class SEResidualUnit(keras.layers.Layer):
    def __init__(
        self, filters, kernel_size, strides=1, se_active=False, squeeze_factor=16, **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_active = se_active
        self.squeeze_factor = squeeze_factor
        self.activation = GeneralizedSwish()
        self.se_unit = SEUnit(squeeze_factor)
        self.shortcut_connection = []
        self.feed_forward = [
            DefaultConv2D(filters=filters, kernel_size=kernel_size, strides=strides),
            layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters=filters, kernel_size=kernel_size),
            layers.BatchNormalization(),
        ]

    def build(self, input_shape):
        if not self.filters == input_shape[-1] or self.strides > 1:
            self.shortcut_connection = [
                DefaultConv2D(filters=self.filters, kernel_size=1, strides=self.strides),
                layers.BatchNormalization(),
            ]
        super().build(input_shape)

    def call(self, inputs):
        X, shortcut_X = inputs, inputs
        for layer in self.feed_forward:
            X = layer(X)
        for layer in self.shortcut_connection:
            shortcut_X = layer(shortcut_X)
        residual_unit_output = self.activation(X + shortcut_X)
        if self.se_active:
            return self.se_unit(residual_unit_output) + shortcut_X
        return residual_unit_output

    def get_config(self):
        return super().get_config() | {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_active": self.se_active,
            "squeeze_factor": self.squeeze_factor,
        }


def parse_example(serialized_example, /):
    feature_descr = {
        "image": tf.io.VarLenFeature(dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_descr)
    return tf.io.decode_jpeg(example["image"].values[0], channels=3), example["label"][0]


def get_dataset_from_tfrecord(
    filename,
    /,
    compression="GZIP",
    batch_size=32,
    cache=True,
    shuffle=False,
    shuffle_buffer_size=256,
    seed=None,
):
    dataset = tf.data.TFRecordDataset(filename, compression_type=compression)
    dataset = dataset.map(parse_example, num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size, num_parallel_calls=AUTOTUNE)
    return dataset.prefetch(AUTOTUNE)


def prepare_image(filename, /):
    image = tf.io.decode_jpeg(tf.io.read_file(str(filename)), channels=3)
    resizing = layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True)
    image_prepared = tf.cast(resizing(image), dtype=tf.uint8)
    return tf.reshape(image_prepared, (1, 224, 224, 3))
