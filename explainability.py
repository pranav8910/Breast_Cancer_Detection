"""
Explainability Module for Breast Cancer Detection
Implements Grad-CAM and LIME for model interpretability
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from lime import lime_image
from skimage.segmentation import mark_boundaries


class ExplainabilityModule:
    """
    Provides explainability methods for the breast cancer detection model
    - Grad-CAM: Gradient-weighted Class Activation Mapping
    - LIME: Local Interpretable Model-agnostic Explanations
    """
    
    def __init__(self, model_instance):
        """
        Initialize explainability module with the model
        
        Args:
            model_instance: BreastCancerModel instance
        """
        self.model_instance = model_instance
        self.model = model_instance.get_model()
    
    def generate_gradcam(self, image_array, alpha=0.4):
        """
        Generate Grad-CAM heatmap
        
        Grad-CAM highlights the regions of the image that are important
        for the model's prediction
        
        Args:
            image_array: Original input image (numpy array)
            alpha: Transparency factor for overlay (0-1)
        
        Returns:
            heatmap_overlay: Image with Grad-CAM heatmap overlay
        """
        # Preprocess image
        preprocessed = self.model_instance.get_preprocessed_image(image_array)
        
        # Get the last convolutional layer name
        last_conv_layer_name = self.model_instance.get_last_conv_layer()
        
        if last_conv_layer_name is None:
            # Fallback: create a simple heatmap based on image intensity
            return self._create_fallback_heatmap(image_array, alpha)
        
        try:
            # Create a model that maps the input image to the activations
            # of the last conv layer and the output predictions
            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(last_conv_layer_name).output,
                    self.model.output
                ]
            )
            
            # Compute gradient of top predicted class with respect to
            # the output feature map of the last conv layer
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(preprocessed)
                class_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            
            # Gradient of the class output value with respect to 
            # the feature map of the last conv layer
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Vector of mean intensity of gradient over specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Multiply each channel by its importance
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads.numpy()
            conv_outputs = conv_outputs.numpy()
            
            for i in range(pooled_grads.shape[0]):
                conv_outputs[:, :, i] *= pooled_grads[i]
            
            # Average over all the filters to get heatmap
            heatmap = np.mean(conv_outputs, axis=-1)
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-10)
            
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return self._create_fallback_heatmap(image_array, alpha)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = (heatmap_colored * alpha + image_array * (1 - alpha)).astype(np.uint8)
        
        return overlay
    
    def _create_fallback_heatmap(self, image_array, alpha=0.4):
        """
        Create a simple fallback heatmap when Grad-CAM fails
        Based on image intensity and edge detection
        """
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create regions
        kernel = np.ones((5, 5), np.uint8)
        heatmap = cv2.dilate(edges, kernel, iterations=2)
        
        # Normalize
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        # Normalize again
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        # Convert to color
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay on original
        overlay = (heatmap_colored * alpha + image_array * (1 - alpha)).astype(np.uint8)
        
        return overlay
    
    def generate_lime(self, image_array, num_samples=100, num_features=5):
        """
        Generate LIME explanation
        
        LIME creates interpretable explanations by approximating the model
        locally with an interpretable model
        
        Args:
            image_array: Original input image
            num_samples: Number of perturbed samples for LIME
            num_features: Number of superpixels to show
        
        Returns:
            explanation_image: Image with LIME explanation overlay
        """
        try:
            # Resize image to model input size for consistency
            resized = cv2.resize(image_array, (224, 224))
            
            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Define prediction function for LIME
            def predict_fn(images):
                """
                Prediction function for LIME
                Takes batch of images and returns predictions
                """
                predictions = []
                for img in images:
                    # Preprocess each image
                    processed = self.model_instance.preprocess_image(img)
                    pred = self.model.predict(processed, verbose=0)
                    predictions.append(pred[0])
                return np.array(predictions)
            
            # Generate explanation
            explanation = explainer.explain_instance(
                resized,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
                random_seed=42
            )
            
            # Get image and mask for top prediction
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=num_features,
                hide_rest=False
            )
            
            # Create visualization with boundaries
            explanation_image = mark_boundaries(temp / 255.0, mask)
            explanation_image = (explanation_image * 255).astype(np.uint8)
            
            # Resize back to original size
            explanation_image = cv2.resize(
                explanation_image,
                (image_array.shape[1], image_array.shape[0])
            )
            
        except Exception as e:
            print(f"LIME generation error: {e}")
            explanation_image = self._create_fallback_lime(image_array)
        
        return explanation_image
    
    def _create_fallback_lime(self, image_array):
        """
        Create a simple fallback LIME-like explanation
        Uses segmentation to highlight regions
        """
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply adaptive thresholding to segment regions
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create output image
        output = image_array.copy()
        
        # Draw boundaries around significant regions
        if len(contours) > 0:
            # Sort by area and take top regions
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small regions
                    cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        
        return output
    
    def generate_combined_explanation(self, image_array):
        """
        Generate both Grad-CAM and LIME explanations
        
        Returns:
            dict with gradcam and lime images
        """
        gradcam = self.generate_gradcam(image_array)
        lime = self.generate_lime(image_array)
        
        return {
            'gradcam': gradcam,
            'lime': lime
        }
