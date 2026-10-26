import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras.losses import MeanSquaredError

def polinomial(x):
    return 5 * x**3 - 10 * x**2 -20 * x + 10
    
x = np.linspace(-20,20,5000).reshape(-1,1)
y = polinomial(x)

x_scaler = MinMaxScaler(feature_range=(-1,1))
y_scaler = MinMaxScaler(feature_range=(-1,1))

x_norm = x_scaler.fit_transform(x)
y_norm = y_scaler.fit_transform(y)

plt.figure(figsize=(8,5))
plt.plot(x,y,color='blue', linewidth=2)
plt.title("Ploynomial Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
    

plt.figure(figsize=(8,5))
plt.plot(x_norm,y_norm,color='blue', linewidth=2)
plt.title("Ploynomial Function")
plt.xlabel("x_normalized")
plt.ylabel("y_normalized")
plt.show()


X_train, X_temp, y_train, y_temp = train_test_split(
   x_norm,
   y_norm,
   test_size = 0.2,
   random_state = 42
)

X_val, X_test, y_val, y_test = train_test_split(
   X_temp,
   y_temp,
   test_size = 0.5,
   random_state = 42
)

model = tf.keras.Sequential([
   Input(shape=(1,)),
   Dense(32, activation='relu'),
   Dense(64, activation='relu'),
   Dense(128, activation='relu'),
   Dense(1)
])

model.compile(
   optimizer='adam',
   loss='mse',
   metrics=['mae']
)
   
history = model.fit(
   X_train,
   y_train,
   validation_data=(X_val,y_val),
   epochs=100,
   batch_size=32,
   verbose=1
)


plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title("Training and Validation Loss over Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.show()


loss, mae = model.evaluate(X_test, y_test, verbose=1)

print("MSE : ",loss)
print("MAE : ",mae)

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
print(f'R2 Score on Test Set: {r2:.6f}')

plt.figure(figsize=(8,5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title("Training MAE vs Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()


