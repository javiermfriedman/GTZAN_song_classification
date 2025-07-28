"""
GTZAN Music Genre Classification - Model Utilities
================================================

This module provides utility functions for model analysis, visualization, and
evaluation. It includes tools for plotting training history, analyzing model
performance, and generating visualizations for model interpretation.

Author: [Your Name]
Date: [Date]
Version: 1.0

Dependencies:
    - matplotlib.pyplot: For creating plots and visualizations
    - numpy: For numerical operations (if needed for future extensions)

Functions:
    - plot_history: Creates training/validation accuracy and loss plots

Usage:
    from model_utils import plot_history
    
    # Plot training history
    plot_history(history, "path/to/save/plot.png")
    
    # The function will display the plot and save it to the specified path
"""

from matplotlib import pyplot as plt

def plot_history(hist, path_to_save_plot):
    """Plots the accuracy and loss for a model over the course of all epochs
    
    Parameters:
        hist (keras history object): The recorded history of model.fit() to be plotted
    """
    fig, axs = plt.subplots(2, 1, figsize=(8,7))
    fig.tight_layout(pad=2)
    
    # Accuracy subplot
    axs[0].plot(hist.history["acc"], c='green', label="Training Accuracy") 
    axs[0].plot(hist.history["val_acc"], c='blue', label="Validation Accuracy")  
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")
    axs[0].set_ylim(0.0, 1.0)  # Set y-axis from 0.0 to 1.0
    
    # Error subplot
    axs[1].plot(hist.history["loss"], c='green', label="Training Loss")
    axs[1].plot(hist.history["val_loss"], c='blue', label="Validation Loss")    
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss")

    plt.savefig(path_to_save_plot)
    
    plt.show()
