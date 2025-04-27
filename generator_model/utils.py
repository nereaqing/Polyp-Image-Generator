import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses, filename="loss_history.png"):
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plotting training and validation loss curves
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')

    # Adding a legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Save the plot as a file
    plt.savefig(filename)

    # Optionally, display the plot
    plt.show()

    print(f"Plot saved as {filename}")
