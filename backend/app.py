from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import os

# Import model
from model import PlantDiseaseCNN

app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = []

def format_class_name(name):
    """Format class name for display (handles mixed formats)"""
    if "__" in name:
        parts = name.split("__")
        if len(parts) == 3:
            # Format: Pepper__bell__Bacterial_spot
            plant, bell, disease = parts
            return f"{plant.replace('_', ' ')} Bell - {disease.replace('_', ' ')}"
        elif len(parts) == 2:
            # Format: Potato__Early_blight
            plant, disease = parts
            return f"{plant.replace('_', ' ')} - {disease.replace('_', ' ')}"
        else:
            return name.replace('_', ' ')
    elif "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0].replace('_', ' ')} - {parts[1].replace('_', ' ')}"
    else:
        return name

def parse_class_name(name):
    """Parse class name into components"""
    if "__" in name:
        parts = name.split("__")
        if len(parts) == 3:
            # Format: Pepper__bell__Bacterial_spot
            plant, bell, disease = parts
            return {
                'plant': plant.replace('_', ' '),
                'type': 'Bell' if 'bell' in bell.lower() else '',
                'disease': disease.replace('_', ' '),
                'full_name': name
            }
        elif len(parts) == 2:
            # Format: Potato__Early_blight
            plant, disease = parts
            return {
                'plant': plant.replace('_', ' '),
                'type': '',
                'disease': disease.replace('_', ' '),
                'full_name': name
            }
    elif "_" in name:
        parts = name.split("_", 1)
        return {
            'plant': parts[0].replace('_', ' '),
            'type': '',
            'disease': parts[1].replace('_', ' '),
            'full_name': name
        }
    else:
        return {
            'plant': name,
            'type': '',
            'disease': '',
            'full_name': name
        }

def load_model():
    global model, class_names
    
    print("=" * 60)
    print("PLANT DISEASE DETECTION API - PlantVillage Dataset")
    print("=" * 60)
    
    try:
        # Load model weights
        state_dict = torch.load('best_model.pth', map_location='cpu')
        print("✓ Model weights loaded")
        
        # Get number of classes
        num_classes = state_dict['fc2.weight'].shape[0]
        print(f"✓ Number of classes: {num_classes}")
        
        # Create and load model
        model = PlantDiseaseCNN(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ Model architecture loaded")
        
        # Load the EXACT class names you provided
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
            
            if len(class_names) == num_classes:
                print(f"✓ Loaded {num_classes} class names from file")
            else:
                print(f"✗ Error: File has {len(class_names)} classes, model expects {num_classes}")
                return False
        else:
            print("✗ Error: class_names.json not found")
            return False
        
        # Show all classes
        print("\nDETECTION CLASSES:")
        print("-" * 40)
        for i, name in enumerate(class_names):
            display = format_class_name(name)
            print(f"  Class {i}: {display}")
        
        print("\n" + "=" * 60)
        print("✓ MODEL LOADED SUCCESSFULLY!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Image transformations (must match your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Based on your model architecture
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/')
def home():
    return jsonify({
        'message': 'Plant Disease Detection API',
        'dataset': 'PlantVillage (8 selected classes)',
        'status': 'running',
        'model_loaded': model is not None,
        'num_classes': len(class_names) if class_names else 0,
        'endpoints': {
            'GET /': 'This info page',
            'GET /health': 'Health check',
            'GET /classes': 'List all detection classes',
            'POST /predict': 'Upload image for disease detection'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_loaded': len(class_names) if class_names else 0
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Return all class names with formatted display names"""
    if not class_names:
        return jsonify({'error': 'Classes not loaded'}), 500
    
    formatted_classes = []
    for name in class_names:
        parsed = parse_class_name(name)
        display = format_class_name(name)
        
        formatted_classes.append({
            'id': name,
            'display': display,
            'plant': parsed['plant'],
            'disease': parsed['disease'],
            'type': parsed['type'],
            'is_healthy': 'healthy' in name.lower()
        })
    
    return jsonify({
        'count': len(class_names),
        'classes': formatted_classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded', 'success': False}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'success': False}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'success': False}), 400
    
    # Check file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if '.' not in file.filename:
        return jsonify({'error': 'Invalid file', 'success': False}), 400
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'File type .{ext} not allowed', 'success': False}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get results
        predicted_idx = predicted_idx.item()
        predicted_class = class_names[predicted_idx]
        confidence_score = confidence.item() * 100
        
        # Get top 3 predictions
        top3_conf, top3_idx = torch.topk(probabilities, 3)
        top_predictions = []
        for i in range(3):
            idx = top3_idx[0][i].item()
            conf = top3_conf[0][i].item() * 100
            class_name = class_names[idx]
            parsed = parse_class_name(class_name)
            
            top_predictions.append({
                'class': class_name,
                'display_name': format_class_name(class_name),
                'plant': parsed['plant'],
                'disease': parsed['disease'],
                'type': parsed['type'],
                'confidence': round(conf, 2),
                'is_healthy': 'healthy' in class_name.lower()
            })
        
        # Parse the predicted class
        parsed = parse_class_name(predicted_class)
        display_name = format_class_name(predicted_class)
        
        # Determine plant type
        if "Pepper" in predicted_class:
            plant_type = "Pepper Bell"
        elif "Potato" in predicted_class:
            plant_type = "Potato"
        elif "Tomato" in predicted_class:
            plant_type = "Tomato"
        else:
            plant_type = "Unknown"
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'display_name': display_name,
            'plant': parsed['plant'],
            'disease': parsed['disease'],
            'plant_type': plant_type,
            'confidence': round(confidence_score, 2),
            'class_index': predicted_idx,
            'is_healthy': 'healthy' in predicted_class.lower(),
            'top_predictions': top_predictions,
            'advice': get_advice(predicted_class, confidence_score),
            'message': 'Analysis complete. Upload a clear photo of plant leaves for best results.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

def get_advice(predicted_class, confidence):
    """Provide advice based on prediction"""
    if 'healthy' in predicted_class.lower():
        return "Your plant appears healthy! Continue regular care and monitoring."
    elif 'bacterial_spot' in predicted_class.lower():
        return "Bacterial spot detected. Remove infected leaves, avoid overhead watering, and consider copper-based fungicides."
    elif 'early_blight' in predicted_class.lower():
        return "Early blight detected. Remove infected leaves, improve air circulation, and apply fungicide containing chlorothalonil."
    elif 'late_blight' in predicted_class.lower():
        return "Late blight detected. Remove and destroy infected plants immediately. Apply copper-based fungicides preventatively."
    elif 'spider_mites' in predicted_class.lower():
        return "Spider mites detected. Spray plants with water to dislodge mites, use insecticidal soap, or neem oil."
    else:
        return "Disease detected. Isolate the plant, remove infected parts, and consider appropriate fungicide treatment."

if __name__ == '__main__':
    print("\nInitializing Plant Disease Detection System...")
    print("Loading trained model for 8 PlantVillage classes...\n")
    
    if load_model():
        print("\n" + "=" * 60)
        print("🚀 API READY: http://localhost:5000")
        print("=" * 60)
        print("\nEndpoints:")
        print("  • http://localhost:5000/          - API info")
        print("  • http://localhost:5000/classes   - List all classes")
        print("  • http://localhost:5000/predict   - Upload image (POST)")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED to load model. Check errors above.")
        print("=" * 60)