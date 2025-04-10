import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision.transforms import (Compose, Normalize, RandomRotation, RandomAdjustSharpness, Resize, ToTensor)
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from transformers import set_seed
import seaborn as sns

class TumorClassifier:
    def __init__(self, data_dir, model_name="google/vit-base-patch16-224-in21k"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = None
        self.train_data = None
        self.val_data = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.trainer = None
        self.label2id = {0.0: 0, 1.0: 1}
        self.id2label = {0.0: "no_tumor", 1.0: "tumor"}
        self.transforms = self._define_transforms()

    def _define_transforms(self):
        image_mean, image_std = self.processor.image_mean, self.processor.image_std
        size = self.processor.size["height"]
        normalize = Normalize(mean=image_mean, std=image_std)

        train_transforms = Compose([
            Resize((size, size)),
            RandomRotation(15),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ])

        val_transforms = Compose([
            Resize((size, size)),
            ToTensor(),
            normalize,
        ])
        return {"train": train_transforms, "val": val_transforms}

    def _apply_transforms(self, generator, transform_fn):
        transformed_images = []
        transformed_labels = []
        for images, labels in generator:
            for i in range(len(images)):
                image = Image.fromarray((images[i] * 255).astype('uint8'))
                transformed_image = transform_fn({'image': [image]})['pixel_values'][0]
                transformed_images.append(transformed_image)
                transformed_labels.append(labels[i])
            break
        return np.stack(transformed_images), np.array(transformed_labels)

    def load_data(self, train_dir=None, val_dir=None):
        
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
        )

        train_images, train_labels = self._apply_transforms(train_generator, self.transforms["train"])
        val_images, val_labels = self._apply_transforms(val_generator, self.transforms["val"])

        self.train_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(train_images, train_labels)]
        self.val_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(val_images, val_labels)]

        self.train_dataloader = DataLoader(self.train_data, collate_fn=self._collate_fn, batch_size=4)
        self.val_dataloader = DataLoader(self.val_data, collate_fn=self._collate_fn, batch_size=4)

    def _collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([self.label2id[example["label"]] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def initialize_model(self):
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name, id2label=self.id2label, label2id=self.label2id
        )

    def train(self, output_dir="checkpoints", num_epochs=20, learning_rate=5e-5, args=None):

        set_seed(62)

        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            data_collator=self._collate_fn,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
        )

        self.trainer.train()

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def evaluate(self):
        self.trainer.state.log_history

        # Extract training and validation accuracy from trainer's logs
        train_logs = self.trainer.state.log_history
        eval_accuracy = [log["eval_accuracy"] for log in train_logs if "eval_accuracy" in log]
        eval_loss = [log["eval_loss"] for log in train_logs if "eval_loss" in log]

        # Plotting eval accuracy and eval loss
        fig, ax1 = plt.subplots()

        color = 'blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(eval_accuracy, color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'red'
        ax2.set_ylabel('Loss', color=color)
        ax2.plot(eval_loss, color=color, label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # to prevent overlap of labels
        plt.title('Validation Accuracy and Loss')
        plt.show()

        # Save the trained model and tokenizer
        self.model.save_pretrained("saved_model")
        self.processor.save_pretrained("saved_model")

        # Extract precision, recall, and F1 score from trainer's logs
        eval_precision = [log["eval_precision"] for log in train_logs if "eval_precision" in log]
        eval_recall = [log["eval_recall"] for log in train_logs if "eval_recall" in log]
        eval_f1 = [log["eval_f1"] for log in train_logs if "eval_f1" in log]

        # Plotting precision, recall, and F1 score
        plt.figure(figsize=(10, 6))
        plt.plot(eval_precision, label='Precision', marker='o', color='blue')
        plt.plot(eval_recall, label='Recall', marker='o', color='green')
        plt.plot(eval_f1, label='F1 Score', marker='o', color='red')

        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1 Score Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
        return self.trainer.evaluate()

    def predict(self, test_dir):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
        )
        test_images, test_labels = self._apply_transforms(test_generator, self.transforms["val"])
        test_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(test_images, test_labels)]
        outputs = self.trainer.predict(test_data)
        return outputs, test_images, test_labels
    
    def display_results(self, outputs, test_images, test_labels):
        labels = ["No tumor", "tumor"]
        y_true = outputs.label_ids
        y_pred = outputs.predictions.argmax(1)
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues', xticklabels=labels, yticklabels=labels)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # Generate predictions
        y_pred_classes = outputs.predictions.argmax(axis=1)  # Get predicted class indices

        # Plot images with actual and predicted labels
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 20))
        axes = axes.flatten()

        for index, (image, label) in enumerate(zip(test_images, test_labels)):
            # Convert image tensor to numpy array and denormalize
            image = image.transpose(1, 2, 0) * 0.5 + 0.5  # Convert from (C, H, W) to (H, W, C) and denormalize
            actual_class = "no_tumor" if label == 0 else "tumor"
            predicted_class = self.id2label[y_pred_classes[index]]
            if index >= len(axes):
                break
            # Display the image
            ax = axes[index]
            ax.imshow(image)
            ax.set_title(f'Actual: {actual_class}\nPredict: {predicted_class}', fontsize=10)
            ax.axis('off')

        # Hide any unused subplots
        for ax in axes[len(test_images):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    data_dir = "D:/SE_AI/Sem_4/Deep_Learning/Group_Project/data/processed"
    
    # data_dir = os.path.join("Group_Project", "data", "processed")
    classifier = TumorClassifier(data_dir)
    # define train and validation directory
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    #load data
    classifier.load_data(train_dir, val_dir)

    #initialize model
    classifier.initialize_model()

    # define training arguments
    args = TrainingArguments(
            "checkpoints",
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='logs',
            remove_unused_columns=False,
            report_to="tensorboard",
        )
    
    # Train the model
    classifier.train(args=args)

    # Evaluate the model
    eval_results = classifier.evaluate()
    print("Evaluation Results:", eval_results)

    
    test_dir = os.path.join(data_dir, 'test')

    outputs = classifier.predict(test_dir)

    print(outputs.metrics)
    classifier.display_results(outputs)