import torch
import torch.nn as nn
from flask import Flask, request, render_template
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Define model
class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.fc = nn.Linear(224 * 224, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load model
def load_model():
    model = DummyCNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).convert("L")
            img = transform(img).unsqueeze(0)
            output = model(img)
            pred_class = torch.argmax(output, 1).item()
            prediction = "Pneumonia" if pred_class == 1 else "Normal"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
