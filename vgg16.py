import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

tf.keras.utils.set_random_seed(42)

# Load CIFAR-10.
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# CIFAR-10 labels: airplane=0, automobile=1, truck=9.
def select_classes(x, y):
    y = y.flatten()
    selected = np.isin(y, [0, 1, 9])

    x = x[selected]
    y = y[selected]

    # Change labels 0, 1, 9 into 0, 1, 2.
    y = np.where(y == 9, 2, y)

    return x, y

x_train, y_train = select_classes(x_train, y_train)
x_test, y_test = select_classes(x_test, y_test)

# Keep 20% of training data for validation.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Training:", len(x_train))
print("Validation:", len(x_val))
print("Testing:", len(x_test))

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

# Freeze pretrained layers.
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Apply the preprocessing expected by VGG16.
    layers.Lambda(preprocess_input),

    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Create fresh callbacks for each training stage.
def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5
        )
    ]

history1 = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=get_callbacks()
)

# Evaluate the frozen pretrained model.
loss1, accuracy1 = model.evaluate(x_test, y_test, verbose=0)

base_model.trainable = True

# Train only the last convolution block.
for layer in base_model.layers:
    layer.trainable = layer.name.startswith("block5")

# Recompile with a smaller learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=get_callbacks()
)

loss2, accuracy2 = model.evaluate(x_test, y_test, verbose=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history1.history["loss"], label="Training loss")
plt.plot(history1.history["val_loss"], label="Validation loss")
plt.title("Frozen Pretrained VGG16")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history2.history["loss"], label="Training loss")
plt.plot(history2.history["val_loss"], label="Validation loss")
plt.title("Partially Fine-tuned VGG16")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Before fine-tuning: {accuracy1 * 100:.2f}%")
print(f"After fine-tuning:  {accuracy2 * 100:.2f}%")
