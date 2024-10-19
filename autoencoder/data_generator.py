import os
import cv2
from glob import glob
#
import numpy as np
import tensorflow as tf
#


def preprocess_input(image: np.ndarray, input_size: int = 50) -> np.ndarray:
    """ Grayscale, resize, and 0-1 normalization of `image`.

    Args:
    ---
    - `image`: np.ndarray
        The input image
    - `input_size`: The target size to resize `image` to

    Returns:
    ---
        np.ndarray
            The normalized image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32)
    image /= 255
    return image


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data_dir,
        shuffle=True,
        batch_size=64,
        input_size=224,
        #
        **kwargs
    ):
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_size = batch_size
        #
        self.samples = sorted(
            list(glob(os.path.join(data_dir, "*.jpg"))))

        self.shuffle = shuffle
        self.indices = np.arange(len(self.samples))
        #

        self.on_epoch_end()
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.samples)//self.batch_size

    def __getitem__(self, index):
        i = self.indices[index *
                         self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in i]
        return self.__get_batch(batch)

    def __get_batch(self, batch):
        images = np.zeros(
            (self.batch_size, self.input_size, self.input_size),
            dtype=np.float32
        )

        for local_batch_idx, sample_idx in enumerate(batch):
            image_filepath = self.samples[sample_idx]

            image = cv2.imread(image_filepath)
            image = preprocess_input(
                image,
                input_size=self.input_size
            )

            images[local_batch_idx] = image

        return images

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
