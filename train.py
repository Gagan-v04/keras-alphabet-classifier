import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
import visualkeras


# --- 1. Load and preprocess the data ---
csv_path = "A_Z Handwritten Data.csv"
data = pd.read_csv(csv_path)
data_subset = data.iloc[:300000]

X = data_subset.iloc[:, 1:].values.astype('float32').reshape(-1, 28, 28, 1) / 255.0
y = data_subset.iloc[:, 0].values.astype(int)
Y = to_categorical(y, num_classes=26)

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y, test_size=0.2, random_state=42
)

# --- 2. Define the CNN model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Train the model ---
history = model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_test, Y_test)
)

# Save the model
model.save('cnn_model.h5')


# --- 4. Plot training and validation accuracy/loss ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# --- 5. Visualize model with visualkeras and custom text_callable ---
import visualkeras

def text_callable(layer_index, layer):
    above = bool(layer_index % 2)
    # Get the output shape of the layer
    try:
        output_shape = [x for x in list(layer.output.shape) if x is not None]
    except Exception:
        output_shape = layer.output.shape.as_list()
    if isinstance(output_shape[0], tuple):
        output_shape = list(output_shape[0])
        output_shape = [x for x in output_shape if x is not None]
    output_shape_txt = ""
    for ii in range(len(output_shape)):
        output_shape_txt += str(output_shape[ii])
        if ii < len(output_shape) - 2:
            output_shape_txt += "x"
        if ii == len(output_shape) - 2:
            output_shape_txt += "\n"
    output_shape_txt += f"\n{layer.name}"
    return output_shape_txt, above

visualkeras.layered_view(
    model,
    to_file='cnn_visualkeras.png',
    legend=True,
    text_callable=text_callable
)

print("CNN architecture visualization saved as cnn_visualkeras.png")
