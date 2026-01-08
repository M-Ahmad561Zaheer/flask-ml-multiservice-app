import torch
import torch.nn as nn
import os

# Create directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        # Feature Extraction: Input (3, 128, 128)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Output: (32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: (64, 32, 32)
        )

        # Classifier
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Overfitting se bachne ke liye
            nn.Linear(64, 1),
            nn.Sigmoid() # 0 (Female) or 1 (Male) result ke liye
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def save_model():
    print("Building CNN architecture...")
    model = GenderCNN()
    
    # Save dummy weights for initial setup
    torch.save(model.state_dict(), 'models/gender_model.pth')
    print("✅ Success! Empty model saved as 'models/gender_model.pth'")
    print("Note: Run your training script next to populate these weights with real knowledge.")

if __name__ == "__main__":
    save_model()