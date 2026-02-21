import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,ResNet50,MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class Config:
    DATA_DIR = 'dataset/train'
    TEST_DIR = 'dataset/test'
    MODEL_SAVE_PATH = 'models/potato_disease_model.h5'
    CLASS_NAMES_PATH = 'models/class_names.pkl'
    HISTORY_PATH = 'models/training_history.pkl'

    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42

    CLASS_NAMES = ['Healthy', 'Blackspot_Bruising', 'Soft_Rot', 'Brown_Rot', 'Dry_Rot']

class PotatoDiseaseDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.class_names = config.CLASS_NAMES

    def create_data_generator(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8,1.2],
            fill_mode='nearest',
            validation_split=self.config.VALIDATION_SPLIT
        )

        validation_datagen = ImageDataGenerator(
            rescale =1./255,
            validation_split=self.config.VALIDATION_SPLIT
        )

        train_generator = train_datagen.flow_from_directory(
            self.config.DATA_DIR,
            target_size = (self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size = self.config.BATCH_SIZE,
            class_mode = 'categorical',
            subset='training',
            seed = self.config.RANDOM_SEED,
            shuffle = True
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.config.DATA_DIR,
            target_size = (self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size = self.config.BATCH_SIZE,
            class_mode = 'categorical',
            subset='validation',
            seed = self.config.RANDOM_SEED,
            shuffle = False
        )

        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Classes: {train_generator.class_indices}")

        self.class_indices = train_generator.class_indices
        self.class_names = list(train_generator.class_indices.keys())

        with open(self.config.CLASS_NAMES_PATH, 'wb') as f:
            pickle.dump(self.class_indices,f)

        return train_generator, validation_generator 
    
    def cnn_from_scratch(self):
        model=keras.Sequential([
            layers.Conv2D(32,(3,3), activation='relu', padding='same',
                        input_shape=(self.config.IMG_SIZE,self.config.IMG_SIZE,3)),
            layers.BatchNormalization(),
            layers.Conv2D(32,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2,2),
            layers.Dropout(0.25),

            layers.Conv2D(64,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2,2),
            layers.Dropout(0.25),

            layers.Conv2D(128,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2,2),
            layers.Dropout(0.25),

            layers.Conv2D(256,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2,2),
            layers.Dropout(0.25),


            layers.Flatten(),
            layers.Dense(512,activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256,activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names),activation='softmax')
        ])
        return model

    def transfer_learning_model(self,base_model_name='mobilenet'):
        if base_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False,
                            input_shape=(self.config.IMG_SIZE,self.config.IMG_SIZE,3))
            preprocess = vgg_preprocess
        elif base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                            input_shape=(self.config.IMG_SIZE,self.config.IMG_SIZE,3))
            preprocess = resnet_preprocess
        else:
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                            input_shape=(self.config.IMG_SIZE,self.config.IMG_SIZE,3))
            preprocess = mobile_preprocess

        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        predictions = Dense(len(self.class_names), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        return model, preprocess

    def train_model(self, use_transfer_learning=True, base_model='mobilenet'):
        train_generator, validation_generator = self.create_data_generator()

        if use_transfer_learning:
            self.model, preprocess = self.transfer_learning_model(base_model)
            print(f"Using transfer learning with {base_model} base model")
        else:
            self.model = self.cnn_from_scratch()
            print("Using CNN model built from scratch")

        self.model.compile(
            optimizer = Adam(learning_rate=self.config.LEARNING_RATE),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        self.model.summary()

        callbacks = [
            ModelCheckpoint(
                self.config.MODEL_SAVE_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        steps_per_epoch = train_generator.samples // self.config.BATCH_SIZE
        validation_steps = validation_generator.samples // self.config.BATCH_SIZE

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch = steps_per_epoch,
            epochs = self.config.EPOCHS,
            validation_data = validation_generator,
            validation_steps = validation_steps,
            callbacks = callbacks,
            verbose = 1
        )

        with open(self.config.HISTORY_PATH,'wb') as f:
            pickle.dump(self.history.history, f)
        
        return self.history

    def evaluate_model(self):
        self.model = load_model(self.config.MODEL_SAVE_PATH)

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.config.TEST_DIR,
            target_size=(self.config.IMG_SIZE,self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        loss, accuracy, precision, recall = self.model.evaluate(test_generator)
        f1_score = 2*(precision*recall)/(precision+recall+1e-7)

        print(f'\n Test Results')
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")

        Y_pred = self.model.predict(test_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = test_generator.classes

        print("\n Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        plt.figure(figsize=(10,8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.show()

        return{
            'loss':loss,
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'f1_score':f1_score
        }
    
    def plot_training_history(self):
        with open(self.config.HISTORY_PATH,'rb') as f:
            history = pickle.load(f)

        fig, axes = plt.subplots(1,2,figsize=(15,5))

        axes[0].plot(history['accuracy'], label='Training Accuracy')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['loss'], label='Training Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()

if __name__ == "__main__":
    config=Config()
    detector = PotatoDiseaseDetector(config)
    print("="*50)
    print("Training Potato Disease Detection Model")
    print("="*50)

    # history = detector.train_model(use_transfer_learning=False)

    # history = detector.train_model(use_transfer_learning=True,base_model='mobilenet')

    print("\n="*50)
    print("Evaluating Model")
    print("="*50)
    results = detector.evaluate_model()

    detector.plot_training_history()
    print("\n Training Completed")
    print(f"Model saved to :{config.MODEL_SAVE_PATH}")

