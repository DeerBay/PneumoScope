from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
from src.model import PneumoNet

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "saved_models/best_model_20250113-012235_binary.pth"  # Update with your model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoNet(num_classes=1, use_pretrained=False).to(device)

# Load model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # Extract only model weights
model.eval()
# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    # Save the file temporarily
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)
    
    # Preprocess the image
    image = Image.open(filepath).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_tensor).squeeze()
        prediction = (logits > 0).float().item()  # Binary classification: 1 (Pneumonia), 0 (Normal)
    
    # Cleanup uploaded file
    os.remove(filepath)
    
    # Map prediction to label
    result = "Pneumonia" if prediction == 1 else "Normal"
    return render_template('result.html', result=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

