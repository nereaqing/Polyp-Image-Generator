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
    

def visualize_image(image, mask=None, masked_image=None):
    if mask is not None and masked_image is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Mask (Polyp Region)")
        ax[1].axis('off')
        ax[2].imshow(masked_image)
        ax[2].set_title("Masked Image (Polyp Extracted)")
        ax[2].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.imshow(image)
        ax.set_title("Original Image")
        ax.axis('off')
    plt.show()

