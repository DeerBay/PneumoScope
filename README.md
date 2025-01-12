# Pneumonia detection from X-rays

### Project overview
The primary goal of the project is to build a deep learning-based solution for classifying chest X-ray images into healthy or pneumonia categories in the first step, and then also classifying the subcategory of bacterial or viral pneumonia. This project involves training, evaluating, and fine-tuning a neural network to accurately detect pneumonia, within a project for the AI23 class, at IT-Högskolan in Göteborg.

### How to set up the environment 
1. Clone the repository to your local machine.
2. Create a virtual environment using `conda` or `venv`.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Download the dataset from Kaggle by running the `kaggle_downloader.py` script.

### How to prepare the dataset and run the model
1. Run the `data_loader.py` script to preprocess the dataset.
2. Run the `model.py` script to define the model architecture.
3. Run the `train.py` script to train the model and saving the best performing model.
4. Run the `evaluate.py` script to evaluate the model on the test set.

### How to implement the model on new data
TODO: Add instructions on how to use the trained model to make predictions on new data.

### Results and performance metrics
TODO: We will discuss these things more in detail when we have discussed the final results and evaluations more.





### Brief project report
TODO: Write more as we progress in the project.

#### Introduction/purpose of the project

The primary goal of the project is to build a deep learning-based solution for classifying chest X-ray images into healthy or pneumonia categories in the first step, and then also classifying the subcategory of bacterial or viral pneumonia. This project involves training, evaluating, and fine-tuning a neural network to accurately detect pneumonia, within a project for the AI23 class, at IT-Högskolan in Göteborg.

#### Methods
    Data preprocessing
    Model
    Training and evaluation
    Further optimization and fine-tuning
#### Results

#### Discussion












### Project Folder Structure
```
PneumoScope/
│
├── data/                  # Folder for datasets
│   ├── train/             # Training images
│   ├── test/              # Testing images
│   └── val/               # Validation images (optional)
│
├── notebooks/             # Jupyter notebooks for experiments and EDA
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── src/                   # Source code
│   ├── data_loader.py     # Code to load and preprocess the dataset
│   ├── model.py           # Code to define the model architecture
│   ├── train.py           # Script to train the model
│   ├── evaluate.py        # Script to evaluate the model
│   └── utils.py           # Helper functions
│
├── saved_models/          # Folder to save trained models
│   ├── best_model.h5
│   └── latest_model.h5
│
├── results/               # Folder for storing results and logs
│   ├── plots/             # Training/validation curves, confusion matrices
│   └── logs/              # Training logs
│
├── main.py                # Main script, runs training and evaluation
├── requirements.txt       # List of dependencies
├── reorganize_dataset.py  # Resizes train|val|test sizes
├── README.md              # Project description and instructions
├── .gitignore             # Ignored files/folders
└── LICENSE                # License for the project
```
