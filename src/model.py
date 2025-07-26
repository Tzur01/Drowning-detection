"""
CNN model for drowning detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for drowning detection.
    
    This model processes video frames to classify whether a person
    is drowning or swimming normally.
    """
    
    def __init__(self, num_classes=2):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for 'drowning' and 'normal')
        """
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, ModelConfig.CONV1_CHANNELS, 5)
        self.conv2 = nn.Conv2d(ModelConfig.CONV1_CHANNELS, ModelConfig.CONV2_CHANNELS, 5)
        self.conv3 = nn.Conv2d(ModelConfig.CONV2_CHANNELS, ModelConfig.CONV3_CHANNELS, 3)
        self.conv4 = nn.Conv2d(ModelConfig.CONV3_CHANNELS, ModelConfig.CONV4_CHANNELS, 5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(ModelConfig.CONV4_CHANNELS, ModelConfig.FC1_SIZE)
        self.fc2 = nn.Linear(ModelConfig.FC1_SIZE, num_classes)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Global average pooling to handle variable input sizes
        batch_size, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Make predictions with the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
