9"""
Inference Script for Concrete Crack Detection
Real-time prediction on new images with severity assessment
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Same model architecture as training
class CrackDetectionModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(CrackDetectionModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class CrackDetector:
    def __init__(self, model_path='best_crack_detection_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['No Crack', 'Hairline Crack', 'Medium Crack', 'Severe Crack']
        self.severity_scores = [0, 1, 2, 3]  # Numerical severity
        
        # Color coding for visualization
        self.colors = {
            'No Crack': (0, 255, 0),        # Green
            'Hairline Crack': (255, 255, 0), # Yellow
            'Medium Crack': (255, 165, 0),   # Orange
            'Severe Crack': (255, 0, 0)      # Red
        }
        
        # Load model
        self.model = CrackDetectionModel(num_classes=4)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded successfully from {model_path}")
            print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            print(f"⚠ Model file not found. Using untrained model for demonstration.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single_image(self, image_path):
        """Predict crack type for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        pred_class = self.class_names[predicted.item()]
        confidence = confidence.item() * 100
        severity = self.severity_scores[predicted.item()]
        
        # Get all class probabilities
        all_probs = {self.class_names[i]: probabilities[0][i].item() * 100 
                     for i in range(len(self.class_names))}
        
        return {
            'class': pred_class,
            'confidence': confidence,
            'severity': severity,
            'probabilities': all_probs
        }
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with annotations"""
        result = self.predict_single_image(image_path)
        
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image with prediction
        ax1.imshow(img)
        ax1.axis('off')
        
        # Add text overlay
        pred_text = f"{result['class']}\nConfidence: {result['confidence']:.1f}%\nSeverity: {result['severity']}/3"
        color = self.colors[result['class']]
        color_norm = tuple(c/255 for c in color)
        
        ax1.text(10, 30, pred_text, fontsize=14, color='white', 
                bbox=dict(boxstyle='round', facecolor=color_norm, alpha=0.8),
                weight='bold')
        ax1.set_title('Crack Detection Result', fontsize=16, weight='bold')
        
        # Probability distribution bar chart
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors_bar = [tuple(c/255 for c in self.colors[cls]) for cls in classes]
        
        bars = ax2.barh(classes, probs, color=colors_bar)
        ax2.set_xlabel('Probability (%)', fontsize=12)
        ax2.set_title('Class Probability Distribution', fontsize=16, weight='bold')
        ax2.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to prediction_result.png")
        
        plt.close()
        
        return result
    
    def batch_predict(self, image_folder):
        """Predict on multiple images in a folder"""
        results = []
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n[INFO] Processing {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            result = self.predict_single_image(img_path)
            result['filename'] = img_file
            results.append(result)
            
            print(f"  {img_file}: {result['class']} ({result['confidence']:.1f}%)")
        
        return results
    
    def generate_report(self, results, save_path='inspection_report.txt'):
        """Generate a detailed inspection report"""
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CONCRETE STRUCTURE INSPECTION REPORT\n")
            f.write("AI-Powered Crack Detection System\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary statistics
            total = len(results)
            severity_counts = {name: 0 for name in self.class_names}
            
            for result in results:
                severity_counts[result['class']] += 1
            
            f.write("SUMMARY:\n")
            f.write(f"  Total Images Analyzed: {total}\n")
            f.write(f"  No Crack: {severity_counts['No Crack']} ({severity_counts['No Crack']/total*100:.1f}%)\n")
            f.write(f"  Hairline Cracks: {severity_counts['Hairline Crack']} ({severity_counts['Hairline Crack']/total*100:.1f}%)\n")
            f.write(f"  Medium Cracks: {severity_counts['Medium Crack']} ({severity_counts['Medium Crack']/total*100:.1f}%)\n")
            f.write(f"  Severe Cracks: {severity_counts['Severe Crack']} ({severity_counts['Severe Crack']/total*100:.1f}%)\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            if severity_counts['Severe Crack'] > 0:
                f.write("  ⚠ CRITICAL: Immediate structural assessment required!\n")
                f.write("  → Schedule emergency inspection by structural engineer\n")
            elif severity_counts['Medium Crack'] > 0:
                f.write("  ⚠ WARNING: Monitor and plan repairs within 30 days\n")
            elif severity_counts['Hairline Crack'] > 0:
                f.write("  ℹ INFO: Minor cracks detected. Monitor for progression\n")
            else:
                f.write("  ✓ All Clear: No significant cracks detected\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED RESULTS:\n")
            f.write("=" * 70 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['filename']}\n")
                f.write(f"   Class: {result['class']}\n")
                f.write(f"   Confidence: {result['confidence']:.2f}%\n")
                f.write(f"   Severity Score: {result['severity']}/3\n")
                f.write("\n")
        
        print(f"✓ Inspection report saved to {save_path}")

# Demo and usage examples
def main():
    print("=" * 70)
    print("AI-Powered Concrete Crack Detection - Inference Module")
    print("=" * 70)
    
    # Initialize detector
    detector = CrackDetector()
    
    print("\n[INFO] Detector initialized and ready!")
    print("\nUsage Examples:")
    print("1. Single image prediction:")
    print("   result = detector.predict_single_image('path/to/image.jpg')")
    print("\n2. Visualize prediction:")
    print("   detector.visualize_prediction('path/to/image.jpg', 'output.png')")
    print("\n3. Batch prediction:")
    print("   results = detector.batch_predict('path/to/image/folder')")
    print("\n4. Generate report:")
    print("   detector.generate_report(results, 'report.txt')")
    
    # Example usage (uncomment when you have test images):
    """
    # Single prediction with visualization
    result = detector.visualize_prediction('test_image.jpg', 'prediction_output.png')
    print(f"\nPrediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Severity: {result['severity']}/3")
    
    # Batch prediction
    results = detector.batch_predict('test_images/')
    detector.generate_report(results, 'inspection_report.txt')
    """

if __name__ == '__main__':
    main()