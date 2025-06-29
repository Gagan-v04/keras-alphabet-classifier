import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Load the dataset
csv_path = "A_Z Handwritten Data.csv"
data = pd.read_csv(csv_path)

# Preprocess the data
X = data.iloc[:, 1:].values.astype('float32').reshape(-1, 28, 28, 1) / 255.0
y = data.iloc[:, 0].values.astype(int)

# Split the data (only for test)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = load_model('cnn_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Label mapping: 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'
label_map = [chr(i) for i in range(65, 91)]

def predict_and_plot(num_samples):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    sample_images = X_test[indices]
    sample_labels = y_test[indices]

    preds = model.predict(sample_images)
    pred_classes = np.argmax(preds, axis=1)

    plt.figure(figsize=(3 * num_samples, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {label_map[sample_labels[i]]}\nPred: {label_map[pred_classes[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Just change this variable to test different numbers of images
num_test_images = 10
predict_and_plot(num_test_images)

