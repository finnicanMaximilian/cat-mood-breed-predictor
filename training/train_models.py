import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import splitfolders
import cv2
import numpy as np
import random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gups:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✔ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Get the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to CatMoods folder right next to the script
data_dir = os.path.join(script_dir, "CatMoods")
print(f"Looking for CatMoods folder at: {data_dir}")

# Build full path to CatBreeds folder right next to the script
breed_data_dir = os.path.join(script_dir, "CatBreed")
print(f"Looking for CatBreeds folder:  {breed_data_dir}")

# Split the cat breed folder into Train / Test / Validation folders
cat_breed_split_dir = os.path.join(script_dir, "CatBreedSplit")
splitfolders.ratio(breed_data_dir, output=cat_breed_split_dir, seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None)

# Split the mood folder into Train / Test / Validatio folders
cat_mood_split_dir = os.path.join(script_dir, "CatMoodSplit")
splitfolders.ratio(data_dir, output=cat_mood_split_dir, seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None)

# Defining Parameters for images and batch_size
img_size = (224, 224)
img_height, img_width = 224, 224
batch_size = 16

# Create Checkpoint
checkpoint = ModelCheckpoint(
    "breed_model.h5",         # File to save the model
    monitor="val_accuracy",   # Save based on validation accuracy
    save_best_only=True,      # Only keep the best version
    mode="max",               # Maximize accuracy
    verbose=1                 # Print a message when saving
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

breed_train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

# Data Augmentation for test data (only rescaling)
breed_test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
# Data Augmentation for validation data (only rescaling)
breed_valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
print(f"\n Test and Validation Data has been Augmented \n")

mood_train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.35,
    brightness_range=[0.8,1.2],
    horizontal_flip = True,
    fill_mode = 'nearest'
)
# Data Augmentation for test data (only rescaling)
mood_test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
# Data Augmentation for validation data (only rescaling)
mood_valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
print(f"\n Test and Validation Data has been Augmented \n")


# Breed Train / Test / Val Directories
breed_train_dir = os.path.join(cat_breed_split_dir, 'train')
breed_val_dir = os.path.join(cat_breed_split_dir, 'val')
breed_test_dir = os.path.join(cat_breed_split_dir, 'test')

# Define Breed IMG Parameters Train / Test / Val
breed_train_data = breed_train_datagen.flow_from_directory(
    breed_train_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

print(breed_train_data.class_indices)

breed_test_data = breed_test_datagen.flow_from_directory(
    breed_test_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

breed_valid_data = breed_valid_datagen.flow_from_directory(
    breed_val_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

# Get a batch of images and labels
images, labels = next(breed_valid_data)

#Select a random image from the batch
idx = random.randint(0, images.shape[0] - 1)

# Display the image
plt.imshow(images[idx])
plt.show()

# Mood Train / Test / Val Directories
mood_train_dir = os.path.join(cat_mood_split_dir, 'train')
mood_val_dir = os.path.join(cat_mood_split_dir, 'val')
mood_test_dir = os.path.join(cat_mood_split_dir, 'test')

# Define Mood IMG Parameters Train / Test / Val
mood_train_data = mood_train_datagen.flow_from_directory(
    mood_train_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

mood_test_data = mood_test_datagen.flow_from_directory(
    mood_test_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

mood_valid_data = mood_valid_datagen.flow_from_directory(
    mood_val_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

# Get a batch of images and labels from mood folder
images, labels = next(mood_valid_data)

#Select a random image from the batch
idx = random.randint(0, images.shape[0] - 1)

# Display the image
plt.imshow(images[idx])
plt.show()

# Load ResNet50
from keras.applications.resnet import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# ------------------
# PHASE 1: Freeze Base
# ------------------

# Freeze the convolutional base
base_model.trainable = False
    
# Build Breed ResNet50 model
breed_model_ResNet50 = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(12, activation='softmax')
])

# Compile Model
breed_model_ResNet50.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                             loss = 'categorical_crossentropy',
                             metrics = ['accuracy'])

# Train Model
history_frozen = breed_model_ResNet50.fit(
    breed_train_data,
    epochs=8,
    validation_data = breed_valid_data,
    callbacks=[checkpoint, early_stopping]
    )

# ---------------------------------
# PHASE 2: Fine-tune last 10 layers
# ---------------------------------
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Re-compile with LOW learning rate (super important!)
breed_model_ResNet50.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---- Fine-tune ----
history_finetune = breed_model_ResNet50.fit(
    breed_train_data,
    validation_data=breed_valid_data,
    epochs=15,
    callbacks=[checkpoint, early_stopping]
)

# Save final breed model
breed_model_ResNet50.save("breed_model.h5")




# Build Mood ResNet50 model
mood_model_ResNet50 = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

# Compile Model
mood_model_ResNet50.compile(optimizer='adam',
                             loss = 'categorical_crossentropy',
                             metrics = ['accuracy'])

# Train Model 
mood_model_ResNet50.fit(mood_train_data, epochs=25, validation_data = mood_valid_data)
mood_model_ResNet50.save("mood_model.h5")

#Accuracy Tests
test_loss, test_accuracy = breed_model_ResNet50.evaluate(breed_test_data)
print(f"Overall test accuracy: {test_accuracy*100:.2f}%")

# Reset the generator so we start at the beginning
breed_test_data.reset()

# Get all test images and true labels
all_preds = []
all_labels = []

for i in range(len(breed_test_data)):
    x_batch, y_batch = breed_test_data[i]
    preds = breed_model_ResNet50.predict(x_batch)
    preds_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    
    all_preds.extend(preds_classes)
    all_labels.extend(true_classes)

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
breed_names = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
               'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue','Siamese','Sphynx']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=breed_names)
disp.plot(xticks_rotation='vertical', cmap=plt.cm.Blues)
plt.show()