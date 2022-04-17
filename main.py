import os
import tensorflow as tf

from theia.archs import UNet
from typing import Tuple


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


DATA_FOLDER = f"./data/testcities"


def get_mask_path(image_path: str) -> str:
    path_parts = tf.strings.split(image_path, os.path.sep)

    filename_parts = tf.strings.split(path_parts[-1], "_")

    filename = tf.strings.reduce_join(
        tf.concat([filename_parts[:-1], ["gtFine_color.png"]], 0), separator="_"
    )

    mask_path = tf.strings.reduce_join(
        tf.concat([path_parts[:-2], ["masks", filename]], 0), separator=os.path.sep
    )
    return mask_path


def load_img_mask_from_path(image_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    mask_path = get_mask_path(image_path)

    img = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    img = tf.io.decode_png(img, channels=3)
    mask = tf.io.decode_png(mask, channels=1)

    img = tf.image.resize(img, (128, 256))
    mask = tf.image.resize(mask, (128, 256), method="nearest")

    return img, mask


train_ds = tf.data.Dataset.list_files(f"{DATA_FOLDER}/train/images/*", shuffle=False)
train_ds = train_ds.map(load_img_mask_from_path, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(2)

model = UNet(input_shape=(128, 256, 3), n_classes=1)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

model.keras_model.summary(line_length=130)

model.fit(train_ds, epochs=3)
