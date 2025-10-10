import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
CONFIG = {
    "IMG_SIZE": (224, 224),
    "BATCH_SIZE": 32,
    "NUM_CLASSES": 6,
    "EPOCHS": 25,  # Increased epochs for fine-tuning
    "LEARNING_RATE_HEAD": 0.001,
    "LEARNING_RATE_FINETUNE": 1e-5,
    "TRAIN_PATH": 'data/train',
    "VALIDATION_PATH": 'data/validation',
    "TEST_PATH": 'data/test',
    "MODEL_PATH": 'best_waste_classifier.h5',
    "CLASSES_PATH": 'class_indices.json'
}

def create_data_generators(config):
    """Creates and returns the training, validation, and test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config["TRAIN_PATH"],
        target_size=config["IMG_SIZE"],
        batch_size=config["BATCH_SIZE"],
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        config["VALIDATION_PATH"],
        target_size=config["IMG_SIZE"],
        batch_size=config["BATCH_SIZE"],
        class_mode='categorical',
        shuffle=False
    )
    test_generator = validation_datagen.flow_from_directory(
        config["TEST_PATH"],
        target_size=config["IMG_SIZE"],
        batch_size=config["BATCH_SIZE"],
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, validation_generator, test_generator

def build_model(num_classes, img_size):
    """Builds and returns the MobileNetV2-based model."""
    base_model = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Start with frozen base

    inputs = Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Slightly increased dropout
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model

def plot_history(history, fine_tune_history=None):
    """Plots training and validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    initial_epochs = len(acc)

    if fine_tune_history:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if fine_tune_history:
        plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if fine_tune_history:
        plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main training pipeline."""
    train_generator, validation_generator, test_generator = create_data_generators(CONFIG)

    # Save class indices for the app
    with open(CONFIG["CLASSES_PATH"], 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Class indices saved to {CONFIG['CLASSES_PATH']}: {train_generator.class_indices}")

    model, base_model = build_model(CONFIG["NUM_CLASSES"], CONFIG["IMG_SIZE"])

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["LEARNING_RATE_HEAD"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print("\n--- Training the classification head ---")
    history = model.fit(
        train_generator,
        epochs=10, # Train head for a few epochs
        validation_data=validation_generator,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
    )

    # --- Fine-tuning ---
    print("\n--- Starting Fine-Tuning ---")
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["LEARNING_RATE_FINETUNE"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    checkpoint = ModelCheckpoint(
        CONFIG["MODEL_PATH"],
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    fine_tune_history = model.fit(
        train_generator,
        epochs=CONFIG["EPOCHS"],
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint],
        initial_epoch=history.epoch[-1]
    )

    # Plot training history
    plot_history(history, fine_tune_history)

    # Evaluate the final model on the test set
    print("\n--- Evaluating on Test Set with the best model ---")
    # Keras' restore_best_weights in EarlyStopping will have loaded the best weights
    # Or we can load it explicitly from the checkpoint
    best_model = tf.keras.models.load_model(CONFIG["MODEL_PATH"])
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"\nTraining complete! Best model saved as '{CONFIG['MODEL_PATH']}' 🧠")

if __name__ == '__main__':
    main()
