from pathlib import Path
import argparse

import tensorflow as tf
from tensorflow.keras import layers


DATA_DIR = Path("dataset/train")
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 29
MODEL_PATH = "cat_dog_model.h5"


def make_dataset(subset):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )


def prepare_dataset(dataset, training=False):
    if training:
        dataset = dataset.shuffle(1024, seed=SEED)

    dataset = dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    # app.py already converts images to 0..1, so keep the saved model compatible.
    x = layers.Rescaling(2.0, offset=-1.0)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a cat/dog classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds = prepare_dataset(make_dataset("training"), training=True)
    val_ds = prepare_dataset(make_dataset("validation"))

    model, base_model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    compile_model(model, learning_rate=1e-3)
    first_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_val_accuracy = max(first_history.history.get("val_accuracy", [0.0]))
    model.save(MODEL_PATH)

    if args.fine_tune_epochs <= 0:
        print(f"Model saved to {MODEL_PATH}")
        print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
        return

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-5)
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_val_accuracy = max(
        best_val_accuracy,
        max(fine_history.history.get("val_accuracy", [0.0])),
    )
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
