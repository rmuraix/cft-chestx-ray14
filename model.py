import numpy as np
from tensorflow import keras


def build_model(
    num_classes: int = 14, checkpoint_path: str | None = None, mls: bool = True
):
    """
    Builds and returns a Keras model.

    Parameters:
      num_classes (int): The number of output classes. Default is 14.
      checkpoint_path (str | None): The path to the model checkpoint. Must be specified.
      mls (bool): If True, the base model's layers are frozen and a new Dense layer is added on top. Default is True.

    Returns:
      keras.Model: The constructed Keras model.

    Raises:
      ValueError: If checkpoint_path is not specified.
      NotImplementedError: If mls is False (functionality not implemented yet).
    """
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified")

    base_model = keras.models.load_model(checkpoint_path)

    if mls:
        base_model.trainable = False

        dense_layer = base_model.get_layer("dense_2")
        gap_output = base_model.get_layer("global_average_pooling2d_2").output

        new_dense_layer = keras.layers.Dense(
            num_classes, activation=None, name="logits"
        )
        logits = new_dense_layer(gap_output)
        new_dense_layer.set_weights(dense_layer.get_weights())

        model = keras.models.Model(inputs=base_model.input, outputs=logits)
    else:
        model = base_model

    return model


# For testing
if __name__ == "__main__":
    model = build_model(checkpoint_path="checkpoints/base_model.h5")

    model.compile()
    print(model.summary())

    last_layer = model.layers[-1]

    # Check activation function
    if "activation" in last_layer.get_config():
        print("Activation function:", last_layer.get_config()["activation"])
    else:
        print("No activation function specified in the last layer")

    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    logits = model.predict(dummy_input)
    print("Logits shape:", logits.shape)
