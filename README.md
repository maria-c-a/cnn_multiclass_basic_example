This script creates a convolutional neural network deep learning model that classifies images that are black, white, and grey.
This setup is one of the most basic problems for an ML model to learn.  Typically, I make sure that the model training and inference pipeline work with this dataset before moving onto more complex datasets.
By testing our your model structure on this training set, many bugs can be solved since the outcome is expected
You can determine if your model is learning slowly or performing as expected and troubleshoot the reasons why issues are occuring.
You don't need to go on kaggle and download a complex set of images. 
You can see the demonstration of how to execute inference when you want to use your model.
Many bugs may not be obovious due to the complexity of the problem and the black-box nature of the cnn, so this setup helped me test the pipelines in a more straightforward and simple way.

The script demosntrates basic training and inference by calling these four main steps.

1. Create the training images in an organized directory
  -create a folder called "training" in the current working directory
  -create folders called "white", "grey", and "black" in the training directory
  -generate 1000 (each) of white, grey, and black images and save them in their respective images

2. Compile and train the cnn model on the images in the training directory and save the model
   -defining the training images from the directory just created
   -defining the model structure and compiling it
   -fitting the model to the training images
   -saving the model within the training curve in a saved_models folder to later access for inference

 3. generate images to demonstrate using the model
    -generate 1000 (each) of grey, white, and black images in an inference directory

  4. use the model to sort the images in the inference directory into organized folders
     -access the saved model and load it
     -access the inference directory images
     -run the images through the model
     -based on the prediction for each picture, take action to sort them into respective folders
     
