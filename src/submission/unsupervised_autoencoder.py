import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class BrainTumorAutoencoder:
    def __init__(self, base_paths, image_size=(224, 224), batch_size=32):
        self.base_paths = base_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.data = {}
        self.data_noisy = {}
        self.autoencoder = None
        self.history = None

    def load_data(self):
        for split, path in self.base_paths.items():
            if split == 'test':
                dataset = tf.keras.utils.image_dataset_from_directory(
                    directory=path,
                    labels='inferred',
                    label_mode='binary',
                    color_mode='grayscale',
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    shuffle=True,
                    seed=42
                )
            else:
                dataset = tf.keras.utils.image_dataset_from_directory(
                    directory=path,
                    labels=None,
                    color_mode='grayscale',
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    shuffle=True,
                    seed=42
                )
            dataset = dataset.map(lambda x, y=None: (x / 255.0, y) if y is not None else x / 255.0)
            self.data[split] = dataset
            print(f"{split} - Dataset created")

    def plot_sample_images(self, dataset, num_samples=5):
        iterator = iter(dataset)
        batch = next(iterator)
        images = batch[0] if isinstance(batch, tuple) else batch
        images = images.numpy()
        plt.figure(figsize=(10, 5))
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.title(f"Img {i+1}")
            plt.axis('off')
        plt.show()

    def add_noise(self, images):
        noise_factor = 0.2
        noisy_images = images + noise_factor * tf.random.normal(tf.shape(images))
        return tf.clip_by_value(noisy_images, 0.0, 1.0)

    def prepare_noisy_data(self):
        self.data_noisy = {
            'train': self.data['train'].map(lambda x: (self.add_noise(x), x)),
            'val': self.data['val'].map(lambda x: (self.add_noise(x), x)),
            'test': self.data['test'].map(lambda x, y: (self.add_noise(x), x, y))
        }

    def build_autoencoder(self):
        inputs = tf.keras.Input(shape=(224, 224, 1))
        # Encoder
        e1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(inputs)
        e2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(e1)
        e3 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(e2)
        # Decoder
        d1 = tf.keras.layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2)(e3)
        d2 = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(d1)
        d3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(d2)
        # Output layer
        outputs = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d3)
        self.autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.summary()

    def train_autoencoder(self, epochs=10):
        self.history = self.autoencoder.fit(
            self.data_noisy['train'],
            epochs=epochs,
            validation_data=self.data_noisy['val']
        )

    def evaluate_model(self):
        test_loss = self.autoencoder.evaluate(self.data_noisy['test'], verbose=0)
        print(f"Loss (MSE) in test dataset: {test_loss:.6f}")

    def compute_reconstruction_errors_and_labels(self):
        errors, all_images, y_true = [], [], []
        for noisy_images, original_images, labels in self.data_noisy['test']:
            reconstructions = self.autoencoder.predict(noisy_images, verbose=0)
            batch_errors = np.mean((original_images.numpy() - reconstructions) ** 2, axis=(1, 2, 3))
            errors.extend(batch_errors)
            all_images.extend(original_images.numpy())
            y_true.extend(labels.numpy().flatten())
        return np.array(errors), np.array(all_images), np.array(y_true)

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss', color='red')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Function')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_reconstructions(self, dataset, num_samples=5):
        for noisy_images, original_images in dataset.take(1):
            reconstructions = self.autoencoder.predict(noisy_images)
            plt.figure(figsize=(15, 9))
            for i in range(min(num_samples, noisy_images.shape[0])):
                plt.subplot(3, num_samples, i + 1)
                plt.imshow(original_images[i].numpy().squeeze(), cmap='gray')
                plt.title('Original')
                plt.axis('off')
                plt.subplot(3, num_samples, i + num_samples + 1)
                plt.imshow(noisy_images[i].numpy().squeeze(), cmap='gray')
                plt.title('Noisy')
                plt.axis('off')
                plt.subplot(3, num_samples, i + 2 * num_samples + 1)
                plt.imshow(reconstructions[i].squeeze(), cmap='gray')
                plt.title('Denoised')
                plt.axis('off')
            plt.show()

    def generate_classification_report(self, y_true, y_pred):
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues',
                    xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Autoencoder')
        plt.show()


# Example usage
if __name__ == "__main__":
    base_paths = {
        'train': 'C:/Users/Asus/OneDrive/Documentos/Diego/Estudio/Centennial/Semester_4/Deep_Learning/Project/DeepLearning_for_BrainTumorDetection/data/processed/train/no_tumor',
        'test': 'C:/Users/Asus/OneDrive/Documentos/Diego/Estudio/Centennial/Semester_4/Deep_Learning/Project/DeepLearning_for_BrainTumorDetection/data/processed/test/',
        'val': 'C:/Users/Asus/OneDrive/Documentos/Diego/Estudio/Centennial/Semester_4/Deep_Learning/Project/DeepLearning_for_BrainTumorDetection/data/processed/val/no_tumor'
    }
    autoencoder = BrainTumorAutoencoder(base_paths)
    autoencoder.load_data()
    autoencoder.plot_sample_images(autoencoder.data['train'], num_samples=5)
    autoencoder.prepare_noisy_data()
    autoencoder.build_autoencoder()
    autoencoder.train_autoencoder(epochs=10)
    autoencoder.evaluate_model()

    test_errors, test_images, y_true = autoencoder.compute_reconstruction_errors_and_labels()
    threshold = np.percentile(test_errors, 95)
    y_pred = (test_errors >= threshold).astype(int)

    autoencoder.generate_classification_report(y_true, y_pred)
    autoencoder.plot_confusion_matrix(y_true, y_pred)
    autoencoder.plot_training_history()
    autoencoder.plot_reconstructions(autoencoder.data_noisy['val'])