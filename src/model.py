import torch
import torch.nn as nn
import torchvision.models as models

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        # Input shape: [batch_size, 1, 224, 224]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [batch, 16, 112, 112]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [batch, 32, 56, 56]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [batch, 64, 28, 28]
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # [batch, 32, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [batch, 16, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [batch, 1, 224, 224]
            nn.Tanh() # Output in range [-1, 1] matching normalization
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class HybridCovidModel(nn.Module):
    def __init__(self, freeze_resnet=False):
        super(HybridCovidModel, self).__init__()
        self.autoencoder = ConvAutoencoder()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # ResNet expects 3 channels, but our latent space is [64, 28, 28]
        # Modify ResNet first layer to accept 1 channel (grayscale)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Get in_features of ResNet's fc layer
        num_ftrs = self.resnet.fc.in_features
        # Remove the final fully connected layer from resnet
        self.resnet.fc = nn.Identity() 
        
        # The latent vector from autoencoder is 64 * 28 * 28 = 50176
        # We can use a small adaptive pool to reduce it before concatenation
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Output: [batch, 64, 1, 1]
        
        # Final classifier: ResNet features (512) + pooled AE features (64) = 576
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1) # Binary classification: 0 (non-COVID), 1 (COVID)
        )

    def forward(self, x):
        # Autoencoder branch
        encoded, decoded = self.autoencoder(x)
        
        # Reduce encoded dimensions
        ae_features = self.pool(encoded) # [batch, 64, 1, 1]
        ae_features = ae_features.view(ae_features.size(0), -1) # [batch, 64]
        
        # CNN branch
        resnet_features = self.resnet(x) # [batch, 512]
        
        # Concatenate features
        combined_features = torch.cat((ae_features, resnet_features), dim=1) # [batch, 576]
        
        # Final classification
        out = self.classifier(combined_features)
        return out, decoded

if __name__ == "__main__":
    # Test model graph
    model = HybridCovidModel()
    dummy_input = torch.randn(4, 1, 224, 224) # Batch of 4, 1 channel, 224x224
    output, decoded = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Decoded shape: {decoded.shape}")
