import tensorflow as tf

def load_dataset(path, batch_size=128, img_size=(64, 64)):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    dataset = dataset.map(lambda x, y: (x / 255.0, x / 255.0))
    return dataset
