from pathlib import Path

from res_nn_components import (
    GeneralizedSwish,
    MaxDepthPool2D,
    SEResidualUnit,
    prepare_image,
)
from tensorflow import keras


def get_model(filename, /):
    return keras.models.load_model(
        str(filename),
        custom_objects={
            "GeneralizedSwish": GeneralizedSwish,
            "MaxDepthPool2D": MaxDepthPool2D,
            "SEResidualUnit": SEResidualUnit,
        },
    )


def main():
    image_path = Path(input("Enter Image Filename: "))
    model_path = Path("models/model_2023_10_09_13_21_14.keras")
    print()

    model = get_model(model_path)
    image = prepare_image(image_path)
    positive_class_proba = model.predict(image)[0, 0]

    print()
    print("Good Class Probability: ", f"{positive_class_proba:.5f}")


if __name__ == "__main__":
    main()
