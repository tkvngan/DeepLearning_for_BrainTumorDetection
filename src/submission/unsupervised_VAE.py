import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class BrainTumorVAE:
    def __init__(self, image_size=(224, 224), batch_size=32, latent_dim=8):
        self.image_size = image_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.data = {}
        self.data_noisy = {}
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.history = None

    def load_data(self, base_paths):
        for split, path in base_paths.items():
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

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_vae(self):
        # Encoder
        input_img = Input(shape=(self.image_size[0], self.image_size[1], 1))
        h = Conv2D(16, kernel_size=3, activation='relu', padding='same', strides=2)(input_img)
        enc_output = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=2)(h)
        shape = tf.keras.backend.int_shape(enc_output)
        x = Flatten()(enc_output)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        self.encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()

        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        x = Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
        dec_output = Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)
        self.decoder = Model(latent_inputs, dec_output, name='decoder')
        self.decoder.summary()

        # VAE
        outputs = self.decoder(self.encoder(input_img)[2])
        self.vae = Model(input_img, outputs, name='vae')

        # Loss
        reconst_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.keras.backend.flatten(input_img), tf.keras.backend.flatten(outputs)))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.vae.add_loss(reconst_loss + kl_loss)

        # Compile
        self.vae.compile(optimizer='adam')
        self.vae.summary()

    def train(self, epochs=10):
        self.history = self.vae.fit(
            self.data_noisy['train'],
            epochs=epochs,
            validation_data=self.data_noisy['val']
        )

    def evaluate(self):
        def compute_reconstruction_errors_and_labels(model, dataset):
            errors = []
            all_images = []
            y_true = []
            for noisy_images, original_images, labels in dataset:
                reconstructions = model.predict(noisy_images, verbose=0)
                batch_errors = np.mean((original_images.numpy() - reconstructions) ** 2, axis=(1, 2, 3))
                errors.extend(batch_errors)
                all_images.extend(original_images.numpy())
                y_true.extend(labels.numpy().flatten())
            return np.array(errors), np.array(all_images), np.array(y_true)

        test_errors, test_images, y_true = compute_reconstruction_errors_and_labels(self.vae, self.data_noisy['test'])
        threshold = np.percentile(test_errors, 95)
        y_pred = (test_errors >= threshold).astype(int)

        # Metrics
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues',
                    xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - VAE')
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss', color='red')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Function')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    base_paths = {
        'train': 'C:/Users/Asus/OneDrive/Documentos/Diego/Estudio/Centennial/Semester_4/Deep_Learning/Project/DeepLearning_for_BrainTumorDetection/data/processed/train/no_tumor',
        'test': 'C:/Users/Asus/OneDrive/Documentos/Diego/Estudio/Centennial/Semester_4/Deep_Learning/Project/DeepLearning_for_BrainTumorDetection/data/processed/test/',
        'val': 'C:/Users/Asus/OneDrive/Documentos/Diego/Estudio/Centennial/Semester_4/Deep_Learning/Project/DeepLearning_for_BrainTumorDetection/data/processed/val/no_tumor'
    }

    vae_model = BrainTumorVAE()
    vae_model.load_data(base_paths)
    vae_model.prepare_noisy_data()
    vae_model.build_vae()
    vae_model.train(epochs=10)
    vae_model.evaluate()
    vae_model.plot_loss()