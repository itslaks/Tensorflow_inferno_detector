#importing required packages
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import filters, color, img_as_ubyte
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define paths
base_path = "L:/include/Fire detection using tensorflow"  # Base path for the project
train_dir = os.path.join(base_path, 'clean_images')  # Directory containing training images
test_dir = os.path.join(base_path, 'testimags')  # Directory containing test images

# Image Data Generator for training
train_datagen = ImageDataGenerator(rescale=1/255)  # Rescale images to [0, 1] range
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 pixels
    seed=42,  # Seed for reproducibility
    batch_size=32,  # Number of images to process in each batch
    classes=['0', '1']  # Class labels
)

# Get images and labels
images, labels = next(train_generator)  # Fetch a batch of images and their labels
label = [0 if l[0] == 1 else 1 for l in labels]  # Convert one-hot labels to binary labels

# Image Pre-processing Functions
def gaussianblur(input_image):
    return filters.gaussian(input_image, sigma=1, channel_axis=-1)  # Apply Gaussian blur

def edgesobel(input_image):
    return filters.sobel(input_image)  # Apply Sobel filter for edge detection

def otsu(input_image):
    im = tf.squeeze(input_image)  # Remove single-dimensional entries from the shape of an array
    im_np = im.numpy()  # Convert Tensor to NumPy array
    thresh = threshold_otsu(im_np)  # Apply Otsu's thresholding
    binary_image = im_np > thresh  # Convert to binary image
    return tf.convert_to_tensor(img_as_ubyte(binary_image), dtype=tf.uint8)  # Convert back to Tensor

def normalize(input_images):
    hsv_image = tf.image.rgb_to_hsv(input_images)  # Convert RGB to HSV
    h, s, v = tf.split(hsv_image, num_or_size_splits=3, axis=-1)  # Split into channels
    normalized_s = tf.image.per_image_standardization(s)  # Normalize saturation channel
    normalized_hsv_image = tf.concat([h, normalized_s, v], axis=-1)  # Concatenate channels back
    return tf.image.hsv_to_rgb(normalized_hsv_image)  # Convert back to RGB

def image_preprocessing(input_images):
    normalized_images = normalize(input_images)  # Normalize images
    filtered_images = []
    for image in normalized_images:
        blurred_image = gaussianblur(image)  # Apply Gaussian blur
        sobel_image = edgesobel(blurred_image)  # Apply Sobel filter
        otsu_image = otsu(sobel_image)  # Apply Otsu's thresholding
        filtered_images.append(otsu_image)
    return np.array(filtered_images)

# Pre-process and store filtered images
filtered_images = []
batch_size = 32
for start in range(0, len(images), batch_size):
    end = start + batch_size
    filtered_batch = image_preprocessing(images[start:end])  # Process each batch of images
    filtered_images.extend(filtered_batch)

filtered_images = np.array(filtered_images).reshape(-1, 224, 224, 3)  # Reshape processed images
label = np.array(label).reshape(len(label), 1)  # Reshape labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_images, label, test_size=0.2, random_state=42)  # 80% train, 20% test

# Build the model
def build_model(input_shape):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)  # Pre-trained ResNet50
    base_model.trainable = False  # Freeze the base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),  # Flatten the output
        tf.keras.layers.Dropout(0.2),  # Dropout layer to reduce overfitting
        tf.keras.layers.Dense(2048, activation='relu'),  # Fully connected layer
        tf.keras.layers.Dropout(0.25),  # Dropout layer
        tf.keras.layers.Dense(1024, activation='relu'),  # Fully connected layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
    ])
    return model

model = build_model((224, 224, 3))  # Build the model
model.summary()  # Print model summary

# Compile the model
model.compile(
    loss='binary_crossentropy',  # Binary cross-entropy loss
    optimizer='adam',  # Adam optimizer
    metrics=['accuracy', tf.keras.metrics.Precision()]  # Accuracy and Precision as metrics
)

# Train the model
epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32)  # Train the model

# Evaluate the model
predictions = model.predict(X_test)  # Predict on test set
predicted_labels = [1 if pred >= 0.5 else 0 for pred in predictions]  # Convert predictions to binary labels
accuracy = accuracy_score(y_test, predicted_labels)  # Calculate accuracy
print(f"Accuracy: {accuracy}")

# Function to test on corner cases
def test_on_corner_cases(model, test_dir):
    image_list = []
    og_list = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):  # Check for image files
            im = Image.open(os.path.join(test_dir, filename))  # Open image
            og_list.append(im)  # Store original image
            im_resized = im.resize((224, 224))  # Resize image
            image_list.append(np.array(im_resized))  # Convert to NumPy array

    image_list = np.array(image_list) / 255.0  # Rescale images
    filtered_images = image_preprocessing(image_list)  # Pre-process images

    predictions = model.predict(filtered_images)  # Predict on pre-processed images
    pred_labels = [1 if pred >= 0.5 else 0 for pred in predictions]  # Convert predictions to binary labels

    fig, axes = plt.subplots(len(image_list), 1, figsize=(10, 10))  # Create a subplot
    for i, ax in enumerate(axes):
        ax.imshow(og_list[i])  # Display original image
        ax.set_title(f"Predicted: {'Fire' if pred_labels[i] == 1 else 'No Fire'}")  # Set title with prediction
        ax.axis('off')  # Turn off axis
    plt.show()  # Show plot

test_on_corner_cases(model, test_dir)  # Test the model on corner cases
