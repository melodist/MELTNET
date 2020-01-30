""" NetworkKeras
    Create Network for extracting features
"""

import tensorflow as tf


def create_base_network(image_input_shape, embedding_size, weight_CT=0.5):
    """Create Base Network for extracting features using dual modality
        CAUTION: Input size of CT and PT should be same
    Input
    ______
    image_input_shape: input shape for single modality.
    embedding_size: output size
    weight_CT: rate for CT data. Default value is 0.5

    Output
    ______
    base_network: model object
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

    fcn1_CT = tf.keras.layers.Dense(150, activation='relu')(flat_CT)
    fcn2_CT = tf.keras.layers.Dense(embedding_size * weight_CT, activation='relu')(fcn1_CT)
    fcn2_CT_l2 = tf.keras.backend.l2_normalize(fcn2_CT)

    # Model for PT
    input_PT = tf.keras.Input(shape=image_input_shape)
    reshape_PT = tf.keras.layers.Reshape((17, 17, 1), input_shape=(17 * 17, 1,))(input_PT)

    conv1_PT = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')(reshape_PT)
    norm1_PT = tf.keras.layers.BatchNormalization()(conv1_PT)
    pool1_PT = tf.keras.layers.MaxPooling2D(padding='same')(norm1_PT)

    conv2_PT = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')(pool1_PT)
    norm2_PT = tf.keras.layers.BatchNormalization()(conv2_PT)
    pool2_PT = tf.keras.layers.MaxPooling2D(padding='same')(norm2_PT)
    flat_PT = tf.keras.layers.Flatten()(pool2_PT)

    fcn1_PT = tf.keras.layers.Dense(150, activation='relu')(flat_PT)
    fcn2_PT = tf.keras.layers.Dense(embedding_size * (1 - weight_CT), activation='relu')(fcn1_PT)
    fcn2_PT_l2 = tf.keras.backend.l2_normalize(fcn2_PT)

    # Concatenate the output
    added = tf.keras.layers.concatenate([fcn2_CT_l2, fcn2_PT_l2])

    base_network = tf.keras.Model(inputs=[input_CT, input_PT], outputs=added)
    tf.keras.utils.plot_model(base_network, to_file='base_network_191112.png',
                              show_shapes=True, show_layer_names=True)

    return base_network


def create_autoencoder(image_input_shape):
    """Create autoencoder for extracting features
    Input
    ______
    image_input_shape: input shape

    Output
    ______
    autoencoder: model object
    """
    # Encoding Layers
    input = tf.keras.Input(shape=image_input_shape)
    reshape1 = tf.keras.layers.Reshape((17, 17, 1,), input_shape=(17 * 17, 1,))(input)

    conv1_layer = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')
    conv1 = conv1_layer(reshape1)
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1_layer = tf.keras.layers.MaxPooling2D(padding='same')
    pool1 = pool1_layer(norm1)

    conv2_layer = tf.keras.layers.Conv2D(50, [3, 3], activation='relu')
    conv2 = conv2_layer(pool1)
    norm2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2_layer = tf.keras.layers.MaxPooling2D(padding='same')
    pool2 = pool2_layer(norm2)
    flat = tf.keras.layers.Flatten()(pool2)

    # Hidden Layer
    fcn1 = tf.keras.layers.Dense(150, activation='relu')(flat)
    fcn1_l2 = tf.keras.backend.l2_normalize(fcn1)

    # Decoding Layers
    fcn2 = tf.keras.layers.Dense(450, activation='relu')(fcn1_l2)
    fcn2_l2 = tf.keras.backend.l2_normalize(fcn2)

    reshape2 = tf.keras.layers.Reshape((3, 3, 50,), input_shape=(450,))(fcn2_l2)

    # Output shape : (8, 8, 50, )
    pool3 = tf.keras.layers.UpSampling2D()(reshape2)
    conv3_layer = tf.keras.layers.Conv2DTranspose(filters=50, kernel_size=3, activation='relu')
    conv3 = conv3_layer(pool3)
    norm3 = tf.keras.layers.BatchNormalization()(conv3)

    # Output shape : (18, 18, 50, )
    pool4 = tf.keras.layers.UpSampling2D()(norm3)
    conv4_layer = tf.keras.layers.Conv2DTranspose(filters=50, kernel_size=3, activation='relu')
    conv4 = conv4_layer(pool4)
    norm4 = tf.keras.layers.BatchNormalization()(conv4)

    # No activation
    conv5 = tf.keras.layers.Conv2D(filters=1, kernel_size=2)(norm4)
    flat2 = tf.keras.layers.Flatten()(conv5)

    autoencoder = tf.keras.Model(inputs=input, outputs=flat2)
    tf.keras.utils.plot_model(autoencoder, to_file='autoencoder_191126.png',
                              show_shapes=True, show_layer_names=True)

    return autoencoder


def create_single_network(image_input_shape, embedding_size):
    """Create autoencoder for extracting features
    Input
    ______
    image_input_shape: input shape
    embedding_size: output shape

    Output
    ______
    base_network: model object
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

    fcn1_CT = tf.keras.layers.Dense(150, activation='relu')(flat_CT)
    fcn2_CT = tf.keras.layers.Dense(embedding_size, activation='relu')(fcn1_CT)
    fcn2_CT_l2 = tf.keras.backend.l2_normalize(fcn2_CT)

    base_network = tf.keras.Model(inputs=input_CT, outputs=fcn2_CT_l2)
    tf.keras.utils.plot_model(base_network, to_file='base_network_191128.png',
                              show_shapes=True, show_layer_names=True)

    return base_network
