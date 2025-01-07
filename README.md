# PneumoScope
Detecting and classifying pneumonia (bacterial vs viral) on chest X-ray images.

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
