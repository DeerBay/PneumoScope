# PneumoScope
Detecting and classifying pneumonia (bacterial vs viral) on chest X-ray images.

pneumonia-detection/
│
├── data/                  # Folder for datasets
│   ├── train/             # Training images
│   ├── test/              # Testing images
│   └── val/               # Validation images (optional, can be split from train)
│
├── notebooks/             # Jupyter notebooks for experiments and EDA (optional)
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── src/                   # Source code
│   ├── data_loader.py     # Code to load and preprocess the dataset
│   ├── model.py           # Code to define the model architecture
│   ├── train.py           # Script to train the model
│   ├── evaluate.py        # Script to evaluate the model
│   └── utils.py           # Helper functions (e.g., plotting, metrics calculation)
│
├── saved_models/          # Folder to save trained models
│   ├── best_model.h5
│   └── latest_model.h5
│
├── results/               # Folder for storing results and logs
│   ├── plots/             # Training/validation curves, confusion matrix plots
│   └── logs/              # Training logs (if using a logging tool)
│
├── requirements.txt       # List of dependencies (e.g., ...)
├── README.md              # Project description and instructions
└── .gitignore             # Ignored files/folders (e.g., data, saved models)
