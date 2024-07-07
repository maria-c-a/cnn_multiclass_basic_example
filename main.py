import os
import cv2
import numpy as np

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import pandas as pd
import shutil


def create_training_folder_and_images(training_directory):
    # Define the main directory and subdirectories
    subdirectories = ["black", "white", "gray"]

    # Create the main directory
    os.makedirs(training_directory, exist_ok=True)

    
    # Create the subdirectories
    for subdir in subdirectories:
        os.makedirs(os.path.join(training_directory, subdir), exist_ok=True)

    # Define image size
    image_size = (1000, 1000)  # 1000x1000 pixels

    # Create black, white, and gray images
    for i in range(1000):
        # Create black image
        black_image = np.zeros(image_size, dtype=np.uint8)
        black_image_path = os.path.join(training_directory, "black", f"black_image_{i+1}.jpg")
        cv2.imwrite(black_image_path, black_image)
        
        # Create white image
        white_image = np.ones(image_size, dtype=np.uint8) * 255
        white_image_path = os.path.join(training_directory, "white", f"white_image_{i+1}.jpg")
        cv2.imwrite(white_image_path, white_image)
        
        # Create gray image (127 is the middle value between 0 and 255)
        gray_image = np.ones(image_size, dtype=np.uint8) * 127
        gray_image_path = os.path.join(training_directory, "gray", f"gray_image_{i+1}.jpg")
        cv2.imwrite(gray_image_path, gray_image)

def train_model(training_directory, checkpoint_path):
     ##########creating model############
    #create a folder to save the trained model
    saved_models_directory = os.path.dirname(checkpoint_path)
    # Create directory to save the models
    os.makedirs(saved_models_directory, exist_ok=True)

    # Create ImageDataGenerator instances with validation split
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    # Create data generators for training and validation
    train_generator = datagen.flow_from_directory(
        training_directory,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        training_directory,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    # Train the model with the checkpoint callback
    model.fit(
        train_generator,
        epochs=1,
        validation_data=validation_generator,
        callbacks=[checkpoint]
    )

    return checkpoint_path

def create_gr_bl_wht_images_to_test_inference(inference_directory):
    ### creating and testing inference pipeline###
    #in this inference pipeline, we want to move images to their respective folder of their classification

    #first we are going to generate images of white, black, and gray images each with 5000 images in a given folder
    #Note, the "flow from directory method looks for folders within the specified directory for each "category"
    #In this case, we will just have one folder with one "category" called inference_images in which the images are actually stored

    #Create folder and images in correct folder structure
    # Define the main directory and subdirectories
    subdirectories = ["inference_images"]

    # Create the main directory
    os.makedirs(inference_directory, exist_ok=True)


    # Create the subdirectories
    for subdir in subdirectories:
        os.makedirs(os.path.join(inference_directory, subdir), exist_ok=True)

    # Define image size
    image_size = (1000, 1000)  # 1000x1000 pixels

    # Create black, white, and gray images
    for i in range(1000):
        # Create black image
        black_image = np.zeros(image_size, dtype=np.uint8)
        black_image_path = os.path.join(inference_directory, "inference_images", f"black_image_{i+1000}.jpg")
        cv2.imwrite(black_image_path, black_image)
        
        # Create white image
        white_image = np.ones(image_size, dtype=np.uint8) * 255
        white_image_path = os.path.join(inference_directory, "inference_images", f"white_image_{i+1000}.jpg")
        cv2.imwrite(white_image_path, white_image)
        
        # Create gray image (127 is the middle value between 0 and 255)
        gray_image = np.ones(image_size, dtype=np.uint8) * 127
        gray_image_path = os.path.join(inference_directory, "inference_images", f"gray_image_{i+1000}.jpg")
        cv2.imwrite(gray_image_path, gray_image)

def use_model(inference_directory, checkpoint_path):
    # Load the best model in the saved model folder 
    model = tf.keras.models.load_model(checkpoint_path)
    print("loaded")

    #model.load_weights(checkpoint_path)
    inference_directory = f"inference_pipeline\inference"

    
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    print("datagen created")


    inference_generator = datagen.flow_from_directory(
    inference_directory,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
    )
    

    # Get validation file names
    inference_file_names = inference_generator.filenames

    # Get the predictions
    predictions = model.predict(inference_generator)

    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)

    
    

    # Print validation file names and their predictions
    for file_name, pred_class in zip(inference_file_names, predicted_classes):
        print(f"File: {file_name}, Predicted Class: {pred_class}")
        file_path = os.path.join(inference_directory, file_name)
        destination_folder = os.path.join(f'inference_pipeline\organized_images', str(pred_class))
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(file_path, destination_folder)

    

if __name__ == "__main__":

    #name of folder to create to create and store training images that are black, grey, and white
    training_directory = "training"
    
    #call function to training folder of white, black, and gray images. Then generate images in those folders
    create_training_folder_and_images(training_directory)
    #use training directory (folder) of images to train a cnn model to classify them

    checkpoint_path = os.path.join("saved_models", "best_model.keras")
    
    train_model(training_directory, checkpoint_path) #checkpoint path = where the model is saved

    #use the model to categorize images into respective folders

    #first we will create new grey, white, and black images in a folder
    inference_directory = f"inference_pipeline\inference"
    create_gr_bl_wht_images_to_test_inference(inference_directory)
    
    #use model to predict on generated images and sort them into folder 

    use_model(inference_directory, checkpoint_path)

        
