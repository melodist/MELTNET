import tensorflow as tf


def create_base_network(image_input_shape, embedding_size):
    """Create Base Network for extracting features

    """
    # Model for CT
    input_CT = tf.keras.Input(shape=image_input_shape)
    reshape_CT = tf.keras.layers.Reshape((17, 17, 1,), input_shape=(17 * 17, 1,))(input_CT)

    conv1_CT_layer = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')
    conv1_CT = conv1_CT_layer(reshape_CT)
    norm1_CT = tf.keras.layers.BatchNormalization()(conv1_CT)
    pool1_CT_layer = tf.keras.layers.MaxPooling2D(padding='same')
    pool1_CT = pool1_CT_layer(norm1_CT)

    conv2_CT_layer = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')
    conv2_CT = conv2_CT_layer(pool1_CT)
    norm2_CT = tf.keras.layers.BatchNormalization()(conv2_CT)
    pool2_CT_layer = tf.keras.layers.MaxPooling2D(padding='same')
    pool2_CT = pool2_CT_layer(norm2_CT)
    flat_CT = tf.keras.layers.Flatten()(pool2_CT)

    # Model for PT
    input_PT = tf.keras.Input(shape=image_input_shape)
    reshape_PT = tf.keras.layers.Reshape((17, 17, 1), input_shape=(17 * 17, 1,))(input_PT)

    conv1_PT = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')(reshape_PT)
    norm1_PT = tf.keras.layers.BatchNormalization()(conv1_PT)
    pool1_PT = tf.keras.layers.MaxPooling2D(padding='same')(conv1_PT)

    conv2_PT = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')(pool1_PT)
    norm2_PT = tf.keras.layers.BatchNormalization()(conv2_PT)
    pool2_PT = tf.keras.layers.MaxPooling2D(padding='same')(norm2_PT)
    flat_PT = tf.keras.layers.Flatten()(pool2_PT)

    # Model for FCN
    added = tf.keras.layers.concatenate([flat_CT, flat_PT])
    fcn1 = tf.keras.layers.Dense(450, activation='relu')(added)
    fcn2 = tf.keras.layers.Dense(embedding_size, activation='relu')(fcn1)
    fcn2_l2 = tf.keras.backend.l2_normalize(fcn2)

    base_network = tf.keras.Model(inputs=[input_CT, input_PT], outputs=fcn2_l2)
    tf.keras.utils.plot_model(base_network, to_file='base_network.png',
                              show_shapes=True, show_layer_names=True)

    return base_network
