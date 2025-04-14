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
DISPLAY_PLOTS = True

# Set file directory paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MODEL_PATH = f"{ROOT_DIR}/models/EfficientNet_model.h5"

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
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
        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Validation Accuracy', color='blue')
        ax1.plot(self.history.history['val_accuracy'], color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Loss', color='red')
        ax2.plot(self.history.history['val_loss'], color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title('EfficientNet - Validation Accuracy and Loss')
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


    def plot_epoch_metrics(self):
        """
        Calculates and plots training precision, recall, and F1 score over epochs.
        Requires `self.history.history['accuracy']` and `self.history.history['loss']` to be populated.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        import math

        precision_vals = []
        recall_vals = []
        f1_vals = []

        for epoch in range(len(self.history.history['accuracy'])):
            y_true_epoch = []
            y_pred_epoch = []
            for images, labels in self.train_data:
                predictions = self.model.predict(images, verbose=0)
                pred_classes = (predictions > 0.5).astype(int).flatten()
                y_pred_epoch.extend(pred_classes)
                y_true_epoch.extend(labels.numpy())
                break  # one batch for faster estimation

            precision_vals.append(precision_score(y_true_epoch, y_pred_epoch))
            recall_vals.append(recall_score(y_true_epoch, y_pred_epoch))
            f1_vals.append(f1_score(y_true_epoch, y_pred_epoch))

        epochs = range(len(precision_vals))
        plt.figure()
        plt.plot(epochs, precision_vals, 'bo-', label='Training Precision')
        plt.plot(epochs, recall_vals, 'go-', label='Training Recall')
        plt.plot(epochs, f1_vals, 'ro-', label='Training F1 Score')
        plt.title('Precision, Recall, and F1 Score Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        if DISPLAY_PLOTS:
            plt.show()
        plt.savefig(f"{ROOT_DIR}/results/sota/efficient_net_metrics_over_epochs.png")


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


    def create_no_sample_found_image(self, text: str = "No False Predictions"):
        blank_image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        cv2.putText(blank_image, text, (12, 118), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        return blank_image

    def plot_sample_predictions(self, num_images=12):
        samples = []
        index = 0
        for image, label in self.test_data.unbatch():
            samples.append((image.numpy().astype('uint8'), int(label.numpy().item()), self.y_pred_classes[index]))
            index += 1

        picked_sample_indexes = set()

        def pick_sample(label_int, pred_label):
            for index, sample in enumerate(samples):
                image, label, pred = sample
                if label == label_int and pred == pred_label and index not in picked_sample_indexes:
                    picked_sample_indexes.add(index)
                    return image, label, pred
            return self.create_no_sample_found_image(), label_int, pred_label

        picked_samples = [
            pick_sample(0, 0),
            pick_sample(0, 0),
            pick_sample(0, 0),
            pick_sample(0, 0),
            pick_sample(1, 1),
            pick_sample(1, 1),
            pick_sample(1, 1),
            pick_sample(1, 1),
            pick_sample(0, 1),
            pick_sample(0, 1),
            pick_sample(1, 0),
            pick_sample(1, 0),
        ]

        fig, axes = plt.subplots(nrows=num_images // 4, ncols=4, figsize=(10, 10))

        for i, sample in enumerate(picked_samples):
            image, label_int, pred_label = sample
            ax = axes.flat[i]
            ax.imshow(image)
            label_class_name = self.class_names[label_int]
            pred_class_name = self.class_names[pred_label]
            is_false_positive = label_int == 0 and pred_label == 1
            is_false_negative = label_int == 1 and pred_label == 0
            title = f"\nActual: {label_class_name}\nPredict: {pred_class_name}\n"
            title += \
                "(True Negative)" if label_int == 0 and pred_label == 0 else \
                "(True Positive)" if label_int == 1 and pred_label == 1 else \
                "(False Negative)" if is_false_negative else \
                "(False Positive)" if is_false_positive else ""
            ax.set_title(
                title,
                fontsize=12,
                color='red' if is_false_negative else 'blue' if is_false_positive else 'black'
            )
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{ROOT_DIR}/results/sota/efficient_net_sample_predictions.png")
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

    def plot_sample_prediction_from_file(self, image_path, need_preprocess:bool):
        img = cv2.imread(image_path)
        if need_preprocess:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, self.img_size)  # Resize to target size
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            img = cv2.equalizeHist(img)  # Apply histogram equalization
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = np.expand_dims(img, axis=0)
        file_name = os.path.basename(image_path)
        prediction = self.model.predict(img, verbose=0)
        print(f"{file_name}: {prediction}")
        pred_class = int(prediction > 0.5)

        plt.figure(figsize=(8, 8))
        plt.imshow(img[0])
        plt.title(
            f"Predicted: {self.class_names[pred_class]}\n" +
            f"(Probability of having tumor: {prediction[0][0]:.2f}" + ")\n" +
            f"Image: {file_name}", fontsize=12)
        plt.axis('off')
        plt.show()


    def plot_sample_predictions_from_dir(self, dir_path, need_preprocess:bool = True):
        for file in sorted(os.listdir(dir_path)):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(dir_path, file)
                self.plot_sample_prediction_from_file(image_path, need_preprocess)

    def plot_internet_sample_predictions(self):
        self.plot_sample_predictions_from_dir(f"{ROOT_DIR}/data/tumor_internet/", need_preprocess=True)
        # self.plot_sample_predictions_from_dir(f"{ROOT_DIR}/data/processed/test/no_tumor", need_preprocess=False)
        # self.plot_sample_predictions_from_dir(f"{ROOT_DIR}/data/processed/test/tumor", need_preprocess=False)

if __name__ == "__main__":
    classifier = EfficientNetBinaryClassifier(
        train_dir=f'{ROOT_DIR}/data/processed/train/',
        val_dir=f'{ROOT_DIR}/data/processed/val/',
        test_dir=f'{ROOT_DIR}/data/processed/test/'
    )
    classifier.load_data()
    if os.path.exists(MODEL_PATH):
        print(f"Model loaded from {MODEL_PATH}")
        classifier.load_model(MODEL_PATH)
    else:
        print(f"Model not found at {MODEL_PATH}. Training a new model.")
        classifier.build_model()
        classifier.train_model()
        classifier.fine_tune_model()
        classifier.plot_performance()
        classifier.plot_epoch_metrics()
        classifier.save_model(MODEL_PATH)

    classifier.evaluate_model()
    classifier.generate_predictions()
    classifier.plot_confusion_matrix()
    classifier.plot_sample_predictions()
    classifier.plot_internet_sample_predictions()
