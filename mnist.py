import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display first 5 training images
plt.figure(figsize=(10, 2))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.show()

# Normalization 
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Build CNN Model
model = Sequential([

    Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, 3, activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

model.summary()

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Early Stopping Callback
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3,   
    restore_best_weights=True,
    verbose=1
)


# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,

    # Automatically creates validation set
    validation_split=0.2,

    callbacks=[early_stop],

    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Plot Accuracy
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()

# Plot Loss
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()


import numpy as np
# Select one test image
index = 0

# Predict
prediction = model.predict(X_test[index:index+1])
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[index])

random_indices = np.random.choice(len(X_test), 20, replace=True)

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.ravel()

for i, idx in enumerate(random_indices):
    prediction = model.predict(X_test[idx:idx+1], verbose=0)
    predicted_digit = np.argmax(prediction)
    actual_digit = y_test[idx]

    axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')

    color = 'green' if predicted_digit == actual_digit else 'red'
    axes[i].set_title(
        f"Pred: {predicted_digit}\nActual: {actual_digit}",
        color=color,
        fontsize=10
    )
    axes[i].axis('off')

plt.tight_layout()
plt.show()
