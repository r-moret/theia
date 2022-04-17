import theia as th

from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Concatenate,
    Conv2DTranspose,
    CenterCrop,
)
from keras.models import Model
from keras.optimizers import Optimizer
from typing import Any, Callable, Tuple


class UNet(th.Model):
    def __init__(self, input_shape: Tuple[int] = None, n_classes: int = None) -> None:
        self.keras_model = self._keras_unet(input_shape, n_classes)

    def compile(
        self, loss: Callable = None, optimizer: Optimizer = None, *args, **kwargs
    ) -> None:
        self.keras_model.compile(optimizer, loss)

    def fit(self, X: Any, y: Any = None, *args, **kwargs) -> None:
        self.keras_model.fit(X, y, *args, **kwargs)

    def predict(self, X: Any, *args, **kwargs) -> Any:
        return self.keras_model.predict(X, *args, **kwargs)

    @staticmethod
    def _keras_unet(input_shape: Tuple[int] = None, n_classes: int = None) -> Model:
        inputs = Input(shape=input_shape)

        conv_1_1 = Conv2D(64, 3, padding="same", activation="relu")(inputs)
        conv_1_2 = Conv2D(64, 3, padding="same", activation="relu")(conv_1_1)

        pool_1 = MaxPool2D((2, 2), strides=2)(conv_1_2)

        conv_2_1 = Conv2D(128, 3, padding="same", activation="relu")(pool_1)
        conv_2_2 = Conv2D(128, 3, padding="same", activation="relu")(conv_2_1)

        pool_2 = MaxPool2D((2, 2), strides=2)(conv_2_2)

        conv_3_1 = Conv2D(256, 3, padding="same", activation="relu")(pool_2)
        conv_3_2 = Conv2D(256, 3, padding="same", activation="relu")(conv_3_1)

        pool_3 = MaxPool2D((2, 2), strides=2)(conv_3_2)

        conv_4_1 = Conv2D(512, 3, padding="same", activation="relu")(pool_3)
        conv_4_2 = Conv2D(512, 3, padding="same", activation="relu")(conv_4_1)

        pool_4 = MaxPool2D((2, 2), strides=2)(conv_4_2)

        conv_5_1 = Conv2D(1024, 3, padding="same", activation="relu")(pool_4)
        conv_5_2 = Conv2D(1024, 3, padding="same", activation="relu")(conv_5_1)

        up_1 = Conv2DTranspose(512, 2, strides=2, padding="same")(conv_5_2)
        conv_4_2_crop = CenterCrop(up_1.shape[1], up_1.shape[2])(conv_4_2)

        concat_1 = Concatenate()([up_1, conv_4_2_crop])
        conv_6_1 = Conv2D(512, 3, padding="same", activation="relu")(concat_1)
        conv_6_2 = Conv2D(512, 3, padding="same", activation="relu")(conv_6_1)

        up_2 = Conv2DTranspose(256, 2, strides=2, padding="same")(conv_6_2)
        conv_3_2_crop = CenterCrop(up_2.shape[1], up_2.shape[2])(conv_3_2)

        concat_2 = Concatenate()([up_2, conv_3_2_crop])
        conv_7_1 = Conv2D(256, 3, padding="same", activation="relu")(concat_2)
        conv_7_2 = Conv2D(256, 3, padding="same", activation="relu")(conv_7_1)

        up_3 = Conv2DTranspose(128, 2, strides=2, padding="same")(conv_7_2)
        conv_2_2_crop = CenterCrop(up_3.shape[1], up_3.shape[2])(conv_2_2)

        concat_3 = Concatenate()([up_3, conv_2_2_crop])
        conv_8_1 = Conv2D(128, 3, padding="same", activation="relu")(concat_3)
        conv_8_2 = Conv2D(128, 3, padding="same", activation="relu")(conv_8_1)

        up_4 = Conv2DTranspose(64, 2, strides=2, padding="same")(conv_8_2)
        conv_1_2_crop = CenterCrop(up_4.shape[1], up_4.shape[2])(conv_1_2)

        concat_4 = Concatenate()([up_4, conv_1_2_crop])
        conv_9_1 = Conv2D(64, 3, padding="same", activation="relu")(concat_4)
        conv_9_2 = Conv2D(64, 3, padding="same", activation="relu")(conv_9_1)
        conv_9_3 = Conv2D(n_classes, 1, padding="same", activation="softmax")(conv_9_2)

        return Model(inputs, conv_9_3)
