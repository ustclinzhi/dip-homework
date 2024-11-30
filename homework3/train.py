import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork, Discriminator
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

# 初始化模型
generator = FullyConvNetwork()
discriminator = Discriminator()

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.jpg', comparison)

def train_one_epoch(model, discriminator, dataloader, optimizer_g, optimizer_d, criterion_g, criterion_d,  device, epoch, num_epochs):
    model.train()
    discriminator.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # 1. 更新判别器
        if i % 2 == 0:  # 判别器更新频率为2，即每2个batch更新一次
            optimizer_d.zero_grad()
        real_labels = torch.ones(image_semantic.size(0), 1, 8, 8).to(device)
        fake_labels = torch.zeros(image_rgb.size(0), 1, 8, 8).to(device)

        real_outputs = discriminator(image_semantic)
        fake_outputs = discriminator(model(image_rgb).detach())
        
        d_loss_real = criterion_d(real_outputs, real_labels)
        d_loss_fake = criterion_d(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optimizer_d.step()  # 更新判别器参数

        # 2. 更新生成器
        optimizer_g.zero_grad()
        # 结合内容损失
        generated_image = model(image_rgb)
        g_loss_content = criterion_g(generated_image, image_semantic)  # 内容损失
        g_loss_adversarial = criterion_d(discriminator(generated_image), real_labels)  # 对抗损失
        
        g_loss = g_loss_adversarial+g_loss_content  # 总损失为内容损失和对抗损失之和  # 生成的图像应该被判别器认为是真实的

        g_loss.backward()
        optimizer_g.step()  # 更新生成器参数

        running_loss += g_loss.item() + d_loss.item()

      
          

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results-7', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)

    # Define loss functions
    criterion_g = nn.L1Loss()
    criterion_d = nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失
    
    # Define optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

     # 定义学习率调度器
    lr_schedulerg = ExponentialLR(optimizer_g, gamma=0.1)

    lr_schedulerd = ExponentialLR(optimizer_d, gamma=0.1)
    
    lr_schedulerg2 = ExponentialLR(optimizer_g, gamma=0.1)

    lr_schedulerd2 = ExponentialLR(optimizer_d, gamma=0.9)
    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Add a learning rate scheduler for decay (ensure the correct optimizer is used)
    scheduler_g = StepLR(optimizer_g, step_size=200, gamma=0.1)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
            # Define optimizers
       
        
        train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion_g, criterion_d,  device, epoch, num_epochs)
        validate(generator, val_loader, criterion_g, device, epoch, num_epochs)
       
        #lr_schedulerg.step()
        #lr_schedulerd.step()
        

       

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/pix2pix_generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/pix2pix_discriminator_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
