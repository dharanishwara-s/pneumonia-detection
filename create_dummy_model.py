import torch
import torch.nn as nn

# Simple CNN for testing
class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.fc = nn.Linear(224 * 224, 2)  # 2 classes: Normal, Pneumonia

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

# Create model
model = DummyCNN()

# Save only weights
torch.save(model.state_dict(), "model.pth")
print("Dummy model weights saved successfully!")
