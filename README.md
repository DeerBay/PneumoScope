# Pneumonia detection from X-rays

### Project overview
The primary goal of the project is to build a deep learning-based solution for classifying chest X-ray images into healthy or pneumonia categories in the first step, and then also classifying the subcategory of bacterial or viral pneumonia. This project involves training, evaluating, and fine-tuning a neural network to accurately detect pneumonia, within a project for the AI23 class, at IT-Högskolan in Göteborg.

### How to set up the environment 
1. Clone the repository to your local machine.
2. Create a virtual environment using `conda` or `venv`.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Download the dataset from Kaggle by running the `kaggle_downloader.py` script.

### How to prepare the dataset and run the model
1. Run the `reorganize_dataset.py` script to reorganize the dataset into the required structure.
2. Run the `main.py` script to train the model and evaluate it, including saving the best model.







### Recommended to have this structure, each one of us does his own presentation
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
├── requirements.txt       # List of dependencies
├── README.md              # Project description and instructions
├── .gitignore             # Ignored files/folders
└── LICENSE                # License for the project
```
