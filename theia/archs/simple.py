import theia as th

from typing import Tuple, Any, Callable
from keras.models import Model, Sequential
from keras.optimizers import Optimizer
from keras.layers import Input, Dense, Flatten


class SimpleNet(th.Model):
    def __init__(self, input_shape: Tuple[int] = None, n_classes: int = None) -> None:
        self.keras_model = self._keras_unet(input_shape, n_classes)

    def compile(
        self, loss: Callable = None, optimizer: Optimizer = None, *args, **kwargs
    ) -> None:
        self.keras_model.compile(optimizer, loss)

    def fit(self, X: Any, y: Any = None, *args, **kwargs) -> None:
        return self.keras_model.fit(X, y, *args, **kwargs)

    def predict(self, X: Any, *args, **kwargs) -> Any:
        return self.keras_model.predict(X, *args, **kwargs)

    @staticmethod
    def _keras_unet(input_shape: Tuple[int] = None, n_classes: int = None) -> Model:

        inputs = Input(shape=input_shape)

        flatten = Flatten()(inputs)

        dense1 = Dense(200, activation="relu")(flatten)
        dense2 = Dense(200, activation="relu")(dense1)

        outputs = Dense(n_classes, activation="softmax")(dense2)

        return Model(inputs, outputs)
