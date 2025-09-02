import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6 # cardboard, glass, metal, paper, plastic, trash
EPOCHS = 20
LEARNING_RATE = 0.0001

TRAIN_PATH = 'data/train'
VALIDATION_PATH = 'data/validation'
TEST_PATH = 'data/test'

# 1. Data Augmentation and Preprocessing
# For training data, we apply augmentation to make the model more robust.
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

# For validation and test data, we only rescale the pixel values. No augmentation!
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# 2. Create Data Generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Class indices: {train_generator.class_indices}")

# 3. Build the Model using Transfer Learning (MobileNetV2)
# Load the base model without the top classification layer
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model's layers
base_model.trainable = False

# Add our custom classification head
inputs = Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # Regularization
x = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

# 4. Compile the Model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train the Model
# Callbacks to stop training early if performance stops improving
# and to save the best model found during training.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    'best_waste_classifier.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# 6. Evaluate the Model on the Test Set
print("\n--- Evaluating on Test Set ---")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# The best model is already saved as 'best_waste_classifier.h5' by the ModelCheckpoint callback.
print("\nTraining complete! Best model saved as 'best_waste_classifier.h5' 🧠")