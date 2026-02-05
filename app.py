"""
Explainable AI-Based Breast Cancer Detection System
Main Flask Application
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from model import BreastCancerModel
from explainability import ExplainabilityModule

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model and explainability module
model = BreastCancerModel()
explainer = ExplainabilityModule(model)

# Nutrition recommendations based on prediction
NUTRITION_RECOMMENDATIONS = {
    'Benign': {
        'title': 'General Wellness Diet Recommendations',
        'recommendations': [
            'Maintain a balanced diet rich in fruits and vegetables',
            'Include whole grains, lean proteins, and healthy fats',
            'Stay hydrated with plenty of water throughout the day',
            'Limit processed foods and added sugars',
            'Consider foods rich in antioxidants (berries, leafy greens)',
            'Regular exercise and stress management are also important'
        ],
        'foods': [
            'Cruciferous vegetables (broccoli, cauliflower)',
            'Berries (blueberries, strawberries)',
            'Fatty fish (salmon, mackerel)',
            'Nuts and seeds (walnuts, flaxseeds)',
            'Green tea',
            'Tomatoes and leafy greens'
        ]
    },
    'Malignant': {
        'title': 'Supportive Nutrition Guidelines',
        'recommendations': [
            'Focus on nutrient-dense foods to support overall health',
            'Maintain adequate protein intake for tissue repair',
            'Stay well-hydrated and maintain healthy body weight',
            'Consider anti-inflammatory foods',
            'Small, frequent meals if appetite is reduced',
            'Consult with a registered dietitian for personalized advice'
        ],
        'foods': [
            'Lean proteins (chicken, fish, legumes)',
            'Colorful vegetables and fruits',
            'Whole grains (quinoa, brown rice)',
            'Healthy fats (olive oil, avocado)',
            'Turmeric and ginger',
            'Probiotic-rich foods (yogurt, kefir)'
        ],
        'disclaimer': 'This is general information only. Please consult with healthcare professionals for personalized medical and nutritional advice.'
    }
}


@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')


@app.route('/upload')
def upload():
    """Upload page route"""
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction
    Process image in memory without saving to disk
    """
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file format. Please upload JPG or PNG'}), 400
        
        # Read image from memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL image to numpy array for processing
        image_array = np.array(image)
        
        # Make prediction
        prediction, confidence = model.predict(image_array)
        
        # Generate explainability visualizations
        gradcam_image = explainer.generate_gradcam(image_array)
        lime_image = explainer.generate_lime(image_array)
        
        # Convert images to base64 for display
        original_b64 = image_to_base64(image_array)
        gradcam_b64 = image_to_base64(gradcam_image)
        lime_b64 = image_to_base64(lime_image)
        
        # Get nutrition recommendations
        nutrition = NUTRITION_RECOMMENDATIONS[prediction]
        
        # Prepare result data
        result_data = {
            'prediction': prediction,
            'confidence': float(confidence),
            'original_image': original_b64,
            'gradcam_image': gradcam_b64,
            'lime_image': lime_b64,
            'nutrition': nutrition
        }
        
        return render_template('result.html', **result_data)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


def image_to_base64(image_array):
    """
    Convert numpy array to base64 encoded string for display
    Keeps image in memory without saving to disk
    """
    # Convert numpy array to PIL Image
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Breast Cancer Detection System...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
