import os
import random
import sys

import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Control if plots are displayed
DISPLAY_PLOTS = False

# Set file directory paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MODEL_PATH = f"{ROOT_DIR}/models/EfficientNet_model.h5"

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import confusion_matrix, classification_report


class EfficientNetBinaryClassifier:
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
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        for layer in base_model.layers[:100]:  # freeze first 100 layers
            layer.trainable = False

        top_model = models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model = models.Sequential([base_model, top_model])
        print(self.model.summary())

        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def fine_tune_model(self, epochs=10):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=2, verbose=1
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=[reduce_lr]
        )

    def train_model(self, epochs=15):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        class_weights = {0: 1.0, 1: 1.2}  # Adjusted to reduce aggressive penalty

        self.history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=[early_stopping],
            class_weight=class_weights
        )

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_data)
        print(f"EfficientNet - Model Evaluation:")
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
        os.makedirs(f"{ROOT_DIR}/results/sota", exist_ok=True)
        plt.figure()
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='red')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='blue')
        plt.legend()
        plt.title('EfficientNet - Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(f"{ROOT_DIR}/results/sota/efficient_net_accuracy_curve.png")
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

        plt.figure()
        plt.plot(self.history.history['loss'], label='Train Loss', color='red')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='blue')
        plt.legend()
        plt.title('EfficientNet - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(f"{ROOT_DIR}/results/sota/efficient_net_loss_curve.png")
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

    def plot_confusion_matrix(self):
        """
        Displays the confusion matrix with counts and normalized percentages,
        highlights the number of false negatives, and prints the classification report.
        Also saves the confusion matrix plot as an image.
        """
        """
        Displays the confusion matrix and classification report.
        Also prints the number of false negatives (missed tumors).
        """
        cm = confusion_matrix(self.y_true, self.y_pred_classes)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_percentage,
            annot=True, fmt=".2%", cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('EfficientNet - Confusion Matrix')
        plt.savefig(f"{ROOT_DIR}/results/sota/efficient_net_confusion_matrix.png")
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


        report = "EfficientNet - Classification Report:\n" + classification_report(self.y_true, self.y_pred_classes, target_names=self.class_names)
        with open(f"{ROOT_DIR}/results/sota/efficient_net_classification_report.txt", "w") as f:
            f.write(report)
        print("Classification Report:")
        print(report)

    def save_model(self, path):
            self.model.save(path)
            print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def plot_sample_predictions(self, num_images=18):
        fig, axes = plt.subplots(nrows=num_images // 3, ncols=3, figsize=(10, 20))
        index = 0

        for image, label in self.test_data.unbatch():
            if index >= num_images:
                break

            image_np = image.numpy().astype("uint8")
            label_int = int(label.numpy().item())
            pred_label = self.y_pred_classes[index]

            ax = axes.flat[index]
            ax.imshow(image_np)
            ax.set_title(f'EfficientNet - Actual: {self.class_names[label_int]}\nPredict: {self.class_names[pred_label]}', fontsize=10)
            ax.axis('off')

            index += 1

        plt.tight_layout()
        plt.savefig(f"{ROOT_DIR}/results/sota/efficient_net_sample_predictions.png")
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


if __name__ == "__main__":
    classifier = EfficientNetBinaryClassifier(
        train_dir=f'{ROOT_DIR}/data/processed/train/',
        val_dir=f'{ROOT_DIR}/data/processed/val/',
        test_dir=f'{ROOT_DIR}/data/processed/test/'
    )
    classifier.load_data()
    classifier.build_model()
    classifier.train_model()
    classifier.fine_tune_model()
    classifier.plot_performance()
    classifier.save_model(MODEL_PATH)

    classifier.evaluate_model()
    classifier.generate_predictions()
    classifier.plot_confusion_matrix()
    classifier.plot_sample_predictions()
