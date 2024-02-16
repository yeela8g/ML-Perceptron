# ML-Perceptron

This Python program implements a simple Perceptron model for binary classification. The Perceptron algorithm is a type of linear classifier used for supervised learning of binary classifiers. Below is a brief overview of the program structure and functionalities.

### 1. Setup

- This program is designed to run on Google Colab. It mounts your Google Drive to access the necessary files.
- for using the original dataset file use the "dataset.csv" file attached. It's recommended to save it in the Google Drive for using the impowered high performanced GPU of google.
- provide the path name (`FOLDERNAME`) and folder name (`ASSIGNMENTNAME`) where your code and dataset are located in your Google Drive.

### 2. Dataset & Preprocessing

- The program loads a pre-generated dataset provided.
- The original dataset consists of 500 samples, each containing two features and a binary label.
- The data is split into training and testing sets using a ratio of 80:20.

### 3. Perceptron Implementation

- The Perceptron class is implemented with the following methods:
  - `__init__()`: Initializes the Perceptron model with given parameters.
  - `predict()`: Predicts the output label for a given input sample.
  - `evaluate()`: Evaluates the accuracy of the model on a given dataset.
  - `train()`: Trains the Perceptron model using the training data and evaluates performance on both training and testing sets.

### 4. Handling Bias Term

- A column of ones is added to both training and testing features to handle the bias term.
- This ensures that the model doesn't need to learn a separate bias parameter.

### visualization
That is the visualization of the decision boundary that was created by the trained Perceptron:

![image](https://github.com/yeela8g/ML-Perceptron/assets/118124478/105797d7-8cd2-4781-a1fc-6142b74339d1)


### Running the Program

- Execute the provided code in a Python environment that supports the required libraries (such as Google Colab).
- Make sure to set up the correct folder structure and filenames as specified in the code.
- The program will train the Perceptron model and print the iteration number, training accuracy, and testing accuracy for each iteration.
