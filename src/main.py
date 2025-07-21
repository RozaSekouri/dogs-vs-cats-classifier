import kagglehub
import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Configuration Parameters ---
# Define the Kaggle dataset identifier
KAGGLE_DATASET = "salader/dogs-vs-cats"
# Image dimensions for model input
IMG_HEIGHT = 200
IMG_WIDTH = 200
# Batch size for training and validation data generators
BATCH_SIZE = 32
# Maximum number of epochs for training (EarlyStopping will likely stop sooner)
MAX_EPOCHS = 50
# Patience for EarlyStopping (number of epochs to wait for improvement)
EARLY_STOPPING_PATIENCE = 10
# Learning rate for the Adam optimizer
LEARNING_RATE = 0.0001
# Ratio for splitting the downloaded 'train' data into training and validation sets
TRAIN_VALIDATION_SPLIT_RATIO = 0.8

# --- Directory Paths ---
# Directory where kagglehub will download the dataset
# This path is determined by kagglehub.dataset_download()
# The structure inside is usually {path}/dogs-vs-cats/train.zip etc.

# Temporary directory to unzip raw data into
EXTRACTED_RAW_DATA_DIR = "extracted_raw_data"
# Base directory for the organized train/validation dataset
ORGANIZED_DATA_DIR = "cats_and_dogs_organized"
# Directory to save model checkpoints
CHECKPOINT_DIR = "checkpoints"
# Directory to save the final trained model
FINAL_MODEL_SAVE_DIR = "final_model"

# --- Step 1: Loading and Organizing the Dataset ---
def load_and_organize_dataset():
    """
    Downloads the Kaggle dataset, unzips it, and organizes images
    into 'train' and 'validation' directories with 'cats' and 'dogs' subdirectories.
    """
    print("--- Step 1: Loading and Organizing the Dataset ---")

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    # The 'path' here is the directory where kagglehub stores the downloaded content
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Path to downloaded dataset files:", download_path)

    # Construct path to the actual archive files
    # The downloaded content structure from this specific Kaggle dataset is typically:
    # {download_path}/dogs-vs-cats/train.zip
    data_archive_sub_dir = os.path.join(download_path, "dogs-vs-cats")
    train_zip_path = os.path.join(data_archive_sub_dir, "train.zip")

    # Create a temporary directory for raw unzipped data
    os.makedirs(EXTRACTED_RAW_DATA_DIR, exist_ok=True)

    # Unzip train.zip
    print("Unzipping train.zip...")
    # This will create a 'train' folder inside EXTRACTED_RAW_DATA_DIR/train_images_raw
    shutil.unpack_archive(train_zip_path, os.path.join(EXTRACTED_RAW_DATA_DIR, "train_images_raw"))
    print("Train.zip unzipped.")

    # The actual images are usually nested one more level: 'extracted_raw_data/train_images_raw/train'
    source_image_folder = os.path.join(EXTRACTED_RAW_DATA_DIR, "train_images_raw", "train")
    if not os.path.exists(source_image_folder):
        print(f"Warning: Expected raw images at {source_image_folder} but not found.")
        print("Attempting to find images directly in 'train_images_raw' folder...")
        source_image_folder = os.path.join(EXTRACTED_RAW_DATA_DIR, "train_images_raw")
        if not os.path.exists(source_image_folder) or not any(f.lower().endswith('.jpg') for f in os.listdir(source_image_folder)):
             raise FileNotFoundError(f"Could not find source images at {source_image_folder}. Please verify dataset structure after unzipping.")

    # --- Data Organization for ImageDataGenerator ---
    # ImageDataGenerator.flow_from_directory() prefers data split into train/validation/test
    # with subdirectories for each class.
    train_dir = os.path.join(ORGANIZED_DATA_DIR, "train")
    validation_dir = os.path.join(ORGANIZED_DATA_DIR, "validation")

    train_cats_dir = os.path.join(train_dir, "cats")
    train_dogs_dir = os.path.join(train_dir, "dogs")
    validation_cats_dir = os.path.join(validation_dir, "cats")
    validation_dogs_dir = os.path.join(validation_dir, "dogs")

    # Create all necessary directories
    for directory in [train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir]:
        os.makedirs(directory, exist_ok=True)

    # Get all image filenames from the raw unzipped folder
    image_files = [f for f in os.listdir(source_image_folder) if f.lower().endswith('.jpg')]

    # Shuffle for random train/validation split
    random.shuffle(image_files)

    # Determine split point
    train_split_count = int(TRAIN_VALIDATION_SPLIT_RATIO * len(image_files))

    train_files = image_files[:train_split_count]
    validation_files = image_files[train_split_count:]

    print(f"Total images found: {len(image_files)}")
    print(f"Number of images for training: {len(train_files)}")
    print(f"Number of images for validation: {len(validation_files)}")

    # Move images to their respective directories
    print("Organizing images into train and validation folders...")
    for filename in train_files:
        if filename.lower().startswith('cat'):
            shutil.copy(os.path.join(source_image_folder, filename), train_cats_dir)
        elif filename.lower().startswith('dog'):
            shutil.copy(os.path.join(source_image_folder, filename), train_dogs_dir)

    for filename in validation_files:
        if filename.lower().startswith('cat'):
            shutil.copy(os.path.join(source_image_folder, filename), validation_cats_dir)
        elif filename.lower().startswith('dog'):
            shutil.copy(os.path.join(source_image_folder, filename), validation_dogs_dir)
    print("Dataset organized successfully!")

    return train_dir, validation_dir

# --- Step 2: Visualize Input Information & Prepare with ImageDataGenerator ---
def prepare_data_generators(train_dir, validation_dir):
    """
    Visualizes sample images and sets up ImageDataGenerators for training and validation.
    """
    print("\n--- Step 2: Visualize Input Information & Prepare with ImageDataGenerator ---")

    # Visualize first 9 dog images
    print("\nDisplaying sample dog images...")
    plt.figure(figsize=(10, 10))
    actual_dog_images = [os.path.join(train_dir, "dogs", f) for f in os.listdir(os.path.join(train_dir, "dogs")) if f.endswith('.jpg')][:9]
    for i, img_path in enumerate(actual_dog_images):
        ax = plt.subplot(3, 3, i + 1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(f"Dog: {os.path.basename(img_path)}")
        plt.axis("off")
    plt.suptitle("Sample Dog Images (Original Sizes)", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

    # Visualize first 9 cat images
    print("Displaying sample cat images...")
    plt.figure(figsize=(10, 10))
    actual_cat_images = [os.path.join(train_dir, "cats", f) for f in os.listdir(os.path.join(train_dir, "cats")) if f.endswith('.jpg')][:9]
    for i, img_path in enumerate(actual_cat_images):
        ax = plt.subplot(3, 3, i + 1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(f"Cat: {os.path.basename(img_path)}")
        plt.axis("off")
    plt.suptitle("Sample Cat Images (Original Sizes)", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

    # --- Data Augmentation and Preprocessing with ImageDataGenerator ---
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=40,       # Rotate images by up to 40 degrees
        width_shift_range=0.2,   # Shift image horizontally by up to 20% of width
        height_shift_range=0.2,  # Shift image vertically by up to 20% of height
        shear_range=0.2,         # Apply shear transformation
        zoom_range=0.2,          # Apply zoom
        horizontal_flip=True,    # Flip images horizontally
        fill_mode='nearest'      # Fill newly created pixels after rotation/shift
    )

    # Only rescale (normalize) validation data; no augmentation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Use flow_from_directory to load images and labels
    print("\nPreparing training data generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical' # 'categorical' for one-hot encoded labels (e.g., [1,0] for cat, [0,1] for dog)
    )

    print("Preparing validation data generator...")
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Verify class indices assigned by the generator
    print("Class indices (mapping from class name to numerical index):", train_generator.class_indices)

    return train_generator, validation_generator

# --- Step 3: Build an ANN (VGG16-like CNN) ---
def build_model():
    """
    Builds the VGG16-like Convolutional Neural Network model.
    """
    print("\n--- Step 3: Building the VGG16-like ANN Model ---")

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # (200, 200, 3) for color images

    model = Sequential()

    # Block 1: Two Conv2D layers followed by a MaxPool2D layer
    model.add(Conv2D(input_shape=INPUT_SHAPE, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block 2: Two Conv2D layers followed by a MaxPool2D layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block 3: Three Conv2D layers followed by a MaxPool2D layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block 4: Three Conv2D layers followed by a MaxPool2D layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block 5: Three Conv2D layers followed by a MaxPool2D layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Flatten the output of the convolutional layers to feed into dense layers
    model.add(Flatten())

    # Fully Connected (Dense) Layers
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    # Output layer: 2 units for binary classification (cat/dog), with softmax for probabilities
    model.add(Dense(units=2, activation="softmax"))

    # Compile the model: configure the learning process
    # Adam optimizer is chosen for its efficiency
    # 'categorical_crossentropy' is used as the loss function for one-hot encoded labels
    # 'accuracy' is used as the metric to monitor during training
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary() # Print a summary of the model's architecture

    return model

# --- Step 4: Optimize and Train the Model ---
def train_and_evaluate_model(model, train_generator, validation_generator):
    """
    Trains the given model using data generators, with ModelCheckpoint and EarlyStopping callbacks.
    Evaluates the best model on the validation set and plots training history.
    """
    print("\n--- Step 4: Optimizing and Training the Model ---")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, "best_model.keras")

    # ModelCheckpoint callback: Saves the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_accuracy',       # Metric to monitor for improvement
        verbose=1,                    # Log when a model is saved
        save_best_only=True,          # Only save if the current model is better than previous best
        mode='max'                    # We want to maximize validation accuracy
    )

    # EarlyStopping callback: Stops training if validation accuracy doesn't improve for 'patience' epochs
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE, # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        mode='max',
        restore_best_weights=True     # Restore model weights from the epoch with the best monitored value
    )

    callbacks_list = [checkpoint, early_stopping]

    # Calculate steps per epoch for generators
    # .samples gives the total number of images found by flow_from_directory
    steps_per_epoch_train = train_generator.samples // BATCH_SIZE
    steps_per_epoch_validation = validation_generator.samples // BATCH_SIZE

    print(f"\nStarting model training for up to {MAX_EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch_train,
        epochs=MAX_EPOCHS,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_validation,
        callbacks=callbacks_list # Pass the list of callbacks
    )

    # --- Analyze the results ---
    print("\nAnalyzing training results and plotting history...")
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # Load the best model (EarlyStopping with restore_best_weights=True already loads it,
    # but this explicit load ensures we're working with the best saved version)
    best_model = load_model(filepath)
    print(f"\nLoaded best model from: {filepath}")

    # Use the validation set to make predictions and evaluate
    print("\nEvaluating the best model on the validation set...")
    val_loss, val_accuracy = best_model.evaluate(validation_generator, steps=steps_per_epoch_validation)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Example: Make predictions on a small batch from the validation generator
    print("\nMaking predictions on a few validation samples...")
    x_val_sample, y_val_sample = next(validation_generator) # Get a batch of images and labels
    predictions = best_model.predict(x_val_sample)

    # Interpret predictions for the first few samples
    class_names = list(train_generator.class_indices.keys())
    # Invert the dictionary to get index to class name mapping for easier display
    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}

    print("\nSample Predictions:")
    for i in range(min(5, len(x_val_sample))): # Check first 5 samples from the batch
        true_label_idx = np.argmax(y_val_sample[i])
        predicted_label_idx = np.argmax(predictions[i])
        predicted_confidence = predictions[i][predicted_label_idx]

        print(f"  Sample {i+1}:")
        print(f"    True Label: {idx_to_class[true_label_idx]}")
        print(f"    Predicted Label: {idx_to_class[predicted_label_idx]} (Confidence: {predicted_confidence:.2f})")
        # You can optionally print all probabilities if desired:
        # print(f"    Probabilities: {predictions[i]}")

    return best_model, history

# --- Step 5: Save the Model ---
def save_final_model(model):
    """
    Saves the final trained model to a specified directory.
    """
    print("\n--- Step 5: Saving the Model ---")

    os.makedirs(FINAL_MODEL_SAVE_DIR, exist_ok=True)
    final_model_path = os.path.join(FINAL_MODEL_SAVE_DIR, "cat_dog_classifier_final.keras")

    model.save(final_model_path)
    print(f"Final best model saved to: {final_model_path}")

# --- Main Execution Flow ---
if __name__ == "__main__":
    # 1. Load and Organize Dataset
    train_data_dir, validation_data_dir = load_and_organize_dataset()

    # 2. Prepare Data Generators and Visualize
    train_gen, val_gen = prepare_data_generators(train_data_dir, validation_data_dir)

    # 3. Build Model
    cnn_model = build_model()

    # 4. Train and Evaluate Model
    final_trained_model, training_history = train_and_evaluate_model(cnn_model, train_gen, val_gen)

    # 5. Save Final Model
    save_final_model(final_trained_model)

    print("\nAll project steps completed successfully!")