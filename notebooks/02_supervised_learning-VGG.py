
import tensorflow as tf
import logging
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Define dataset paths
TRAIN_DIR = '../data/processed/train/'
TEST_DIR = '../data/processed/test/'
VAL_DIR = '../data/processed/val/'
IMG_SIZE = (224,224)
BATCH_SIZE = 32

# Image preprocessing using image_dataset_from_directory
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    label_mode='binary',
    seed=42
)

class_names = train_data.class_names  # Get class names before transformations
print(f"Class name: {class_names}")

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    label_mode='binary',
    seed=42
)


val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    label_mode='binary',
    seed=42
)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

# VGG16 Model with transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the convolutional base

# Custom classifier
top_model = models.Sequential([
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Combine VGG16 and classifier
model = models.Sequential([base_model, top_model])

print(top_model.summary())
print(model.summary())

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[early_stopping])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Generate predictions and confusion matrix
y_pred = []
y_true = []
for images, labels in test_data:
    predictions = model.predict(images)
    pred_classes = (predictions > 0.5).astype(int).flatten()  # Convert to binary class (0 or 1)
    y_pred.extend(pred_classes)
    y_true.extend(labels.numpy())


plt.plot(history.history['accuracy'], color = 'red', label = 'train')
plt.plot(history.history['val_accuracy'], color = 'blue', label = 'validation')
plt.legend()
plt.show()


plt.plot(history.history['loss'], color = 'red', label = 'train')
plt.plot(history.history['val_loss'], color = 'blue', label = 'validation')
plt.legend()
plt.show()

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues', xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save("VGG_CNN_model.h5")

# Generate predictions
y_pred = model.predict(test_data)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # Convert to binary class (0 or 1)

# Plot images with actual and predicted labels
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 20))
index = 0

for image, label in test_data.unbatch():
    if index >= 18:  # Stop after displaying 18 images
        break

    # Convert image tensor to numpy array
    image = image.numpy().squeeze()  # Remove extra dimensions if grayscale
    label = int(label.numpy().item())  # ✅ Extract scalar value properly

    # Access the current subplot
    ax = axes.flat[index]

    # Display the image
    if image.ndim == 2:  # If grayscale
        ax.imshow(image, cmap='gray')
    else:  # If RGB
        ax.imshow(image.astype("uint8"))

    # Get actual and predicted class names
    actual_class = class_names[label]
    predicted_class = class_names[y_pred_classes[index]]

    # Set title and remove axis
    ax.set_title(f'Actual: {actual_class}\nPredict: {predicted_class}', fontsize=10)
    ax.axis('off')

    index += 1  # Move to the next subplot

plt.tight_layout()
plt.show()



