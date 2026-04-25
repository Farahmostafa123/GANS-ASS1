def build_autoencoder(input_shape, latent_dim=64):
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="latent")(x)
    encoder = Model(inputs, latent, name="encoder")
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16*16*64, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    outputs = decoder(encoder(inputs))
    ae = Model(inputs, outputs, name="autoencoder")
    return ae, encoder, decoder
ae, ae_encoder, ae_decoder = build_autoencoder(input_shape)
