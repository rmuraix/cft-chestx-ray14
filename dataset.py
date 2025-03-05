import os

import pandas as pd
import tensorflow as tf


class DataLoader:
    def __init__(self, csv_path, image_dir, image_size=(224, 224), batch_size=32):
        """
        Args:
            csv_path (str): Path to CSV file containing label information
            image_dir (str): Path of the directory where the image is stored
            image_size (tuple): Image resize size (width, height)
            batch_size (int): batch size
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.df = pd.read_csv(csv_path)

        self.image_index = [fname for fname in self.df["Image Index"]]
        self.image_paths = [
            os.path.join(image_dir, fname) for fname in self.df["Image Index"]
        ]
        self.labels = self.df.iloc[:, 1:].values

    def _load_image(self, image_path):
        """Read and preprocess images."""
        image = tf.io.read_file(image_path)
        # Load PNG images in RGB format
        image = tf.image.decode_png(image, channels=3)
        # Normalized to 0-1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, self.image_size)
        return image

    def data_generator(self):
        """Load images and labels sequentially"""
        for image_path, image_index, label in zip(
            self.image_paths, self.image_index, self.labels
        ):
            image = self._load_image(image_path)
            yield image, image_index, label

    def get_dataset(self, shuffle: bool = False):
        """Create and return a TensorFlow Dataset (dynamically loaded)"""
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(self.labels.shape[1],), dtype=tf.float32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
