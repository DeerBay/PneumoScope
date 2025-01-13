import itertools
from src.train import train_model

def grid_search(data_dir):
    # Define hyperparameters ranges to search
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128, 256],
        'epochs': [20, 30, 40],
        'patience': [5, 8, 10],
        'lr_patience': [2, 3, 4],        
        'lr_factor': [0.1, 0.5]          
    }

    # Generate all combinations
    keys = param_grid.keys()
    combinations = itertools.product(*param_grid.values())

    best_val_acc = 0
    best_params = None
    results = []

    # Try each combination
    for values in combinations:
        params = dict(zip(keys, values))
        print(f"\nTrying parameters: {params}")

        # Train model with current parameters
        model, history = train_model(
            data_dir=data_dir,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            patience=params['patience']
        )

        # Get best validation accuracy from history
        val_acc = max(history['val_acc'])

        results.append({
            'params': params,
            'val_acc': val_acc
        })

        # Update best parameters if needed
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

        print(f"Validation accuracy: {val_acc:.4f}")

    return best_params, results