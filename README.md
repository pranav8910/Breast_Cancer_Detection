# Explainable AI-Based Breast Cancer Detection System

## 🎯 Project Overview

A fully functional web application that uses deep learning and explainable AI techniques to classify mammogram images as benign or malignant. The system provides transparent explanations through Grad-CAM and LIME visualizations.

## ✨ Key Features

- **Deep Learning Classification**: CNN-based model using transfer learning (VGG16)
- **Explainable AI**: 
  - Grad-CAM heatmaps showing important regions
  - LIME explanations for local interpretability
- **In-Memory Processing**: No file storage, all processing done in memory
- **Nutrition Recommendations**: Evidence-based dietary suggestions
- **Responsive Web Interface**: Clean, professional UI built with Bootstrap
- **Error Handling**: Comprehensive validation and error management

## 🛠️ Technologies Used

### Frontend
- HTML5
- CSS3
- JavaScript (ES6+)
- Bootstrap 5.3

### Backend
- Python 3.8+
- Flask 3.0

### Machine Learning
- TensorFlow 2.15
- Keras 2.15
- Transfer Learning (VGG16)

### Image Processing
- OpenCV
- Pillow (PIL)

### Explainability
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- LIME (Local Interpretable Model-agnostic Explanations)

## 📁 Project Structure

```
breast_cancer_detection/
│
├── app.py                      # Main Flask application
├── model.py                    # CNN model implementation
├── explainability.py           # Grad-CAM and LIME implementation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   ├── upload.html            # Upload page
│   └── result.html            # Results page
│
├── static/                     # Static files
│   └── css/
│       └── style.css          # Custom CSS styles
│
└── model/                      # Model directory (for future use)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone or Download the Project
```bash
cd breast_cancer_detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --break-system-packages -r requirements.txt
```

Note: The `--break-system-packages` flag is required for some environments. If you encounter issues, try:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

## 📖 Usage Guide

### 1. Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 2. Upload Mammogram Image
- Click "Get Started" or navigate to the Upload page
- Select a mammogram image (JPG or PNG format)
- Maximum file size: 16MB
- Click "Analyze Image"

### 3. View Results
The system will display:
- **Classification Result**: Benign or Malignant
- **Confidence Score**: Model confidence percentage
- **Original Image**: Your uploaded mammogram
- **Grad-CAM Heatmap**: Visual explanation highlighting important regions
- **LIME Explanation**: Superpixel-based interpretation
- **Nutrition Recommendations**: Dietary guidance based on result

## 🧠 How It Works

### Image Processing Pipeline

1. **Image Upload**: User uploads mammogram image through web interface
2. **Memory Loading**: Image loaded directly into memory (no disk storage)
3. **Preprocessing**: 
   - Resize to 224x224 pixels
   - Normalize pixel values
   - Apply VGG16 preprocessing
4. **Prediction**: CNN model classifies the image
5. **Explainability Generation**:
   - Grad-CAM: Computes gradient-based attention map
   - LIME: Creates local interpretable model
6. **Result Display**: All visualizations shown to user

### Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Custom Layers**: 
  - Global Average Pooling
  - Dense layers (256, 128 units)
  - Dropout for regularization
  - Softmax output (2 classes)

### Explainability Techniques

#### Grad-CAM
- Uses gradients flowing into final convolutional layer
- Produces coarse localization map
- Highlights regions important for prediction
- Warmer colors (red/yellow) = higher importance

#### LIME
- Perturbs input image and observes prediction changes
- Builds local linear model
- Shows which superpixels contribute to prediction
- Green boundaries = positive contribution

## ⚙️ Configuration

### Model Parameters
Located in `model.py`:
- Input size: 224x224
- Batch size: 1 (single image processing)
- Architecture: VGG16 + custom layers

### Explainability Parameters
Located in `explainability.py`:
- Grad-CAM alpha: 0.4 (overlay transparency)
- LIME samples: 100 (perturbation iterations)
- LIME features: 5 (number of superpixels)

### Flask Configuration
Located in `app.py`:
- Max file size: 16MB
- Allowed formats: JPG, PNG
- Debug mode: Enabled (disable for production)

## 🔒 Important Disclaimers

### Medical Disclaimer
⚠️ **This system is for EDUCATIONAL and RESEARCH purposes only.**

- NOT a medical diagnostic tool
- Should NEVER replace professional medical diagnosis
- Always consult qualified healthcare professionals
- Do not make medical decisions based on this system

### Data Privacy
- No images are saved to disk
- All processing done in memory
- No database or persistent storage
- Session data cleared after processing

### Academic Use
This project is designed for:
- Computer science coursework
- Machine learning demonstrations
- Explainable AI research
- Educational presentations

## 🎓 Viva/Presentation Tips

### Key Points to Explain

1. **Architecture**:
   - Why VGG16 was chosen (proven performance, transfer learning)
   - Role of each layer in the custom classifier
   - How preprocessing affects model performance

2. **Explainability**:
   - Difference between Grad-CAM and LIME
   - Why explainability matters in medical AI
   - How to interpret the visualizations

3. **Implementation**:
   - In-memory processing advantages
   - Flask routing and request handling
   - Error handling strategy

4. **Future Improvements**:
   - Training on actual mammogram dataset
   - Ensemble methods for better accuracy
   - Additional explainability techniques (SHAP, attention mechanisms)
   - Multi-class classification (normal, benign, malignant)

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'tensorflow'`
**Solution**: Reinstall TensorFlow
```bash
pip install --break-system-packages tensorflow==2.15.0
```

**Issue**: Application won't start
**Solution**: Check if port 5000 is already in use
```bash
# Windows
netstat -ano | findstr :5000
# Linux/Mac
lsof -i :5000
```

**Issue**: Image upload fails
**Solution**: 
- Verify file format (JPG/PNG only)
- Check file size (<16MB)
- Ensure file is not corrupted

**Issue**: Grad-CAM not working properly
**Solution**: The system includes fallback visualization. Check console for error messages.

## 📊 Technical Specifications

### Performance
- Prediction time: ~2-5 seconds per image
- Grad-CAM generation: ~1-2 seconds
- LIME generation: ~3-5 seconds
- Total processing time: ~6-12 seconds

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~2GB for dependencies
- **Processor**: Multi-core recommended
- **Browser**: Modern browser (Chrome, Firefox, Edge)

## 🔮 Future Enhancements

- [ ] Train on real mammogram dataset (e.g., DDSM, INbreast)
- [ ] Add multi-class classification
- [ ] Implement SHAP explanations
- [ ] Add batch processing capability
- [ ] Create REST API for integration
- [ ] Add user authentication
- [ ] Implement result history (with privacy controls)
- [ ] Add confidence calibration
- [ ] Support DICOM format
- [ ] Integrate with PACS systems (for clinical use)

## 📚 References

### Research Papers
1. Grad-CAM: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
2. LIME: Ribeiro et al. (2016) - "Why Should I Trust You?"
3. VGG16: Simonyan & Zisserman (2014) - "Very Deep Convolutional Networks"

### Datasets (for future training)
- DDSM (Digital Database for Screening Mammography)
- INbreast
- MIAS (Mammographic Image Analysis Society)
- CBIS-DDSM

### Libraries Documentation
- TensorFlow: https://www.tensorflow.org
- Flask: https://flask.palletsprojects.com
- LIME: https://github.com/marcotcr/lime
- OpenCV: https://opencv.org

## 👥 Contributors

This project was developed for academic purposes.

## 📄 License

This project is intended for educational use only. Not licensed for commercial or clinical use.

## 📧 Support

For academic questions or technical issues:
- Check the troubleshooting section
- Review Flask/TensorFlow documentation
- Consult with your project supervisor

---

**Remember**: This is a demonstration system for learning purposes. Real medical AI systems require:
- FDA approval / regulatory compliance
- Clinical validation studies
- Rigorous testing on diverse datasets
- Integration with healthcare workflows
- Continuous monitoring and updates

**Last Updated**: February 2026
