# **To-Do List: Pneumonia Detection and Classification Project**

## **1. Project Setup**
- [X] Create a GitHub repository with a clear name and description.
- [X] Set up a virtual environment (e.g., `uv`, `venv` or `conda`).
- [X] Create a `README.md` file with an initial project description.
- [X] Add a `.gitignore` file (include common Python files, data).
- [X] List project dependencies in (e.g `requirements.txt`).

## **2. Data Preparation**
- [X] Download the dataset from Kaggle. # Sebbe data_loader.py
- [X] Organize the dataset into appropriate directories (`train`, `test`, `val`).
- [X] Explore the dataset (e.g., check for class imbalance, image size). # Julia data_exploration.ipynb
- [X] Perform data preprocessing: 
  - [X] Resize images to a uniform size.
  - [X] Normalize image pixel values.
  - [X] Apply data augmentation (e.g., flipping, rotation, zoom).

## **3. Model Development**
- [X] Choose baseline CNN architecture (e.g., simple CNN). # Ali
- [X] Train a binary classification model (pneumonia vs. no pneumonia).
- [X] Train a ternary classification model (bacterial vs. viral vs. no pneumonia). # Ali
- [X] Experiment with pre-trained models (e.g., ResNet, VGG, Inception).
- [X] Implement transfer learning if using pre-trained models.

## **4. Model Evaluation** # Ali och Julia
- [X] Evaluate models on the test set.
- [X] Compute key metrics:
  - [X] Accuracy
  - [X] Precision, Recall, F1-score, Accuracy
  - [X] ROC-AUC # Ali
- [X] Create and display confusion matrices. # Julia
- [X] Save model performance results for comparison. # Ali

## **5. Optimization** # Sebastian
- [X] Fine-tune hyperparameters (learning rate, batch size, number of epochs). 
- [/] Implement early stopping and learning rate scheduling. # Early stopping check, learning rate scheduler not yet implemented
- [X] Test different optimizers (e.g., Adam, SGD).
- [/] Experiment with different data augmentation strategies. # Implemented but not tested

## **6. Visualization** # Julia
- [X] Plot training and validation loss/accuracy curves. 
- [X] Visualize some correctly and incorrectly classified images. 
- [X] Use Grad-CAM or similar techniques to interpret model predictions. 

## **7. Deployment (Optional)** # Younis
- [X] Convert the trained model to a format suitable for deployment (e.g., PyTorch, ONNX).
- [X] Create a simple web app using Flask or Streamlit for real-time predictions.
- [X] Test the web app with sample X-ray images.

## **8. Documentation** # Younis
- [X] Update the `README.md` file with detailed instructions on:
  - [X] Project overview
  - [X] How to set up the environment
  - [X] How to run the model
  - [X] Results and performance metrics # See analysis_of_models.ipynb
- [X] Add comments to the code for clarity.
- [X] Write a brief project report (as required by our course). # Brief overview added/Younis

## **9. Final Steps**
- [/] Review and clean the code.
- [X] Commit all changes and push to GitHub.
- [X] Share the project link with our course instructor.
