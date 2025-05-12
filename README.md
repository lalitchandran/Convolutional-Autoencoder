# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS
In image processing, noise can significantly degrade visual quality and hinder further analysis. The task is to design a convolutional autoencoder that can automatically learn to remove noise from images. This project utilizes the MNIST handwritten digits dataset as the input and adds Gaussian noise to simulate real-world distortion. The autoencoder is trained to reconstruct clean images from their noisy versions.

Dataset Used:

MNIST: A standard dataset of handwritten digits (0–9), consisting of 60,000 training and 10,000 testing grayscale images of size 28×28.

### STEP 1:

Import Required Libraries
Import necessary Python libraries including torch, torchvision, matplotlib, and numpy for model building, data loading, and visualization.

### STEP 2:

Prepare the Dataset

Load the MNIST dataset using torchvision.datasets.

Normalize the images using transforms.ToTensor().

Split into training and testing sets using DataLoader.


### STEP 3:

Add Gaussian Noise to the Images

Define a function add_noise() that adds random Gaussian noise to the input images.

Ensure pixel values remain in the range [0, 1] using torch.clamp().

### STEP 4:
Build the Convolutional Autoencoder Model

Design an autoencoder class using nn.Module.

The encoder compresses the image into a lower-dimensional representation.

The decoder reconstructs the original image from the compressed form.

Use Conv2d, ConvTranspose2d, ReLU, and Sigmoid layers.

### STEP 5:

Define Loss Function and Optimizer

Use Mean Squared Error (MSE) loss to measure reconstruction accuracy.

Use the Adam optimizer for faster convergence.

### STEP 6:

Train the Model

Loop over multiple epochs.

For each batch, add noise to the images.

Pass the noisy images through the autoencoder.

Compute the loss and update the model weights.

### STEP 7:

Visualize the Results

Select a batch of test images.

Add noise and denoise them using the trained model.

Display the original, noisy, and denoised images using matplotlib.

## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [32, 7, 7]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: S LALIT CHANDRAN")
    print("Register Number: 212223240077")

    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```
### Name: S LALIT CHANDRAN
### Register Number: 212223240077

## OUTPUT

### Model Summary

![image](https://github.com/user-attachments/assets/701ffd5f-24c3-49db-9176-9c2931bf1305)


### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/user-attachments/assets/6033b5fa-d93e-485e-91d9-7839ac37e2a2)



## RESULT
The convolutional autoencoder was successfully trained to remove noise from MNIST images. The denoised outputs closely resembled the original images, demonstrating the model's effectiveness in learning to reconstruct clean images from noisy inputs.
