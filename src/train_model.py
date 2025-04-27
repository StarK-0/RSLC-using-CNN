import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt

# Paths
data_dir = 'E:/Final Year Project/Final Review and Report/signlanguage/dataset'
model_path = 'models/sign_model.h5'
os.makedirs('models', exist_ok=True)

# Parameters
img_size = (128, 128)  # Increased image size for better feature detection
batch_size = 16  # Smaller batch size for better generalization with small dataset
epochs = 50  # More epochs with proper callbacks

# Data Augmentation (more aggressive to handle small dataset)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# Create train and validation generators
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Don't shuffle validation for consistent evaluation
)

# Get class indices and save for prediction
class_indices = train_data.class_indices
class_names = list(class_indices.keys())
print(f"Classes: {class_names}")

# Save class names to a file for the prediction app
with open('models/class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

print(f"✅ Class names saved to models/class_names.txt")

# Transfer learning with MobileNetV2 (better for small datasets)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)

# Freeze the base model layers
base_model.trainable = False

# Create the model with the pre-trained base
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_history.png')
plt.close()

print(f"✅ Training history saved to models/training_history.png")
print(f"✅ Best model saved to {model_path}")

# Fine-tuning phase
print("Starting fine-tuning phase...")
base_model.trainable = True

# Freeze first 100 layers of base model
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with fine-tuning
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Save final model
model.save(model_path.replace('.h5', '_final.h5'))
print(f"✅ Final model saved to {model_path.replace('.h5', '_final.h5')}")