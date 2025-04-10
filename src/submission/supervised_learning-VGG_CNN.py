#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:01:56 2025

@author: yichenhsu
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, classification_report

class VGG16BinaryClassifier:
    def __init__(self, train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        self.model = None
        self.history = None
        self.y_pred_classes = []
        self.y_true = []

    def load_data(self):
        def _load_dataset(path):
            return tf.keras.utils.image_dataset_from_directory(
                path,
                image_size=self.img_size,
                batch_size=self.batch_size,
                color_mode='rgb',
                label_mode='binary',
                seed=42
            )

        self.train_data = _load_dataset(self.train_dir)
        self.val_data = _load_dataset(self.val_dir)
        self.test_data = _load_dataset(self.test_dir)

        self.class_names = self.train_data.class_names
        print(f"Class names: {self.class_names}")

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_data = self.train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_data = self.val_data.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_data = self.test_data.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        base_model.trainable = False

        top_model = models.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model = models.Sequential([base_model, top_model])
        print(self.model.summary())

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, epochs=20):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        self.history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
    def save_model_architecture(self, filepath="model_vgg_cnn.png"):
        tf.keras.utils.plot_model(self.model, to_file=filepath, show_shapes=True)
        print(f"Model architecture saved to {filepath}")

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_data)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        return test_loss, test_acc

    def generate_predictions(self):
        self.y_pred_classes = []
        self.y_true = []

        for images, labels in self.test_data:
            predictions = self.model.predict(images)
            pred_classes = (predictions > 0.5).astype(int).flatten()
            self.y_pred_classes.extend(pred_classes)
            self.y_true.extend(labels.numpy())

    def plot_performance(self):
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='red')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='blue')
        plt.legend()
        plt.title('Model Accuracy')
        plt.show()

        plt.plot(self.history.history['loss'], label='Train Loss', color='red')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='blue')
        plt.legend()
        plt.title('Model Loss')
        plt.show()

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred_classes)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self, path="VGG_CNN_model.h5"):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path="VGG_CNN_model.h5"):
        # Load the model from the specified path
        self.model = load_model(path)
        print(f"Model loaded from {path}")
        
    def plot_sample_predictions(self, num_images=18):
        fig, axes = plt.subplots(nrows=num_images // 3, ncols=3, figsize=(10, 20))
        index = 0

        for image, label in self.test_data.unbatch():
            if index >= num_images:
                break

            image_np = image.numpy().squeeze()
            label_int = int(label.numpy().item())
            pred_label = self.y_pred_classes[index]

            ax = axes.flat[index]
            ax.imshow(image_np.astype("uint8") if image_np.ndim == 3 else image_np, cmap='gray')
            ax.set_title(f'Actual: {self.class_names[label_int]}\nPredict: {self.class_names[pred_label]}', fontsize=10)
            ax.axis('off')

            index += 1

        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    classifier = VGG16BinaryClassifier(
        train_dir='../data/processed/train/',
        val_dir='../data/processed/val/',
        test_dir='../data/processed/test/'
    )
    
    MODEL_PATH = "VGG_CNN_model.h5"
    # Check if the model already exists
    if os.path.exists(MODEL_PATH):
        print(f"Model found at {MODEL_PATH}. Loading the model...")
        # Load the existing model
        classifier.load_data()
        classifier.load_model(MODEL_PATH)
    else:
        print(f"No model found at {MODEL_PATH}. Training a new model...")
        classifier.load_data()
        classifier.build_model()
        classifier.train_model()
        classifier.plot_performance()
        classifier.save_model()
        
    classifier.evaluate_model()
    classifier.generate_predictions()
    classifier.plot_confusion_matrix()
    classifier.plot_sample_predictions()
    