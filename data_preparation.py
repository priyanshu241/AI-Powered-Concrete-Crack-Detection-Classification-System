"""
Data Preparation Script for Concrete Crack Detection
Handles dataset creation, augmentation, and organization
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import random
import json

class CrackDatasetBuilder:
    def __init__(self, output_dir='crack_dataset'):
        self.output_dir = output_dir
        self.classes = ['no_crack', 'hairline', 'medium', 'severe']
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure for the dataset"""
        for split in ['train', 'val', 'test']:
            for cls in self.classes:
                path = os.path.join(self.output_dir, split, cls)
                os.makedirs(path, exist_ok=True)
        print(f"✓ Directory structure created at '{self.output_dir}'")
    
    def generate_synthetic_data(self, num_samples_per_class=200):
        """
        Generate synthetic crack images for demonstration
        In production, replace this with real crack images
        """
        print("\n[INFO] Generating synthetic dataset for demonstration...")
        print("Note: Replace this with real crack images for production use")
        
        for cls_idx, cls in enumerate(self.classes):
            print(f"\nGenerating {cls} images...")
            
            for i in tqdm(range(num_samples_per_class)):
                # Create base concrete texture
                img = self.create_concrete_texture()
                
                # Add cracks based on class
                if cls == 'hairline':
                    img = self.add_hairline_crack(img)
                elif cls == 'medium':
                    img = self.add_medium_crack(img)
                elif cls == 'severe':
                    img = self.add_severe_crack(img)
                # no_crack class gets clean texture
                
                # Add noise and variations
                img = self.add_realistic_effects(img)
                
                # Save to train/val/test split
                split = self.assign_split(i, num_samples_per_class)
                filename = f'{cls}_{i:04d}.jpg'
                save_path = os.path.join(self.output_dir, split, cls, filename)
                cv2.imwrite(save_path, img)
        
        print("\n✓ Synthetic dataset generation complete!")
        self.print_dataset_stats()
    
    def create_concrete_texture(self, size=(512, 512)):
        """Create realistic concrete texture"""
        # Base gray concrete
        base = np.random.randint(120, 160, size, dtype=np.uint8)
        
        # Add texture variation
        noise = np.random.normal(0, 10, size).astype(np.uint8)
        texture = cv2.add(base, noise)
        
        # Add some aggregate-like spots
        for _ in range(random.randint(20, 40)):
            x = random.randint(0, size[0]-20)
            y = random.randint(0, size[1]-20)
            radius = random.randint(3, 8)
            color = random.randint(100, 180)
            cv2.circle(texture, (x, y), radius, int(color), -1)
        
        # Blur slightly for realism
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        
        # Convert to BGR
        texture_bgr = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        
        return texture_bgr
    
    def add_hairline_crack(self, img):
        """Add thin hairline crack"""
        h, w = img.shape[:2]
        
        # Generate crack path
        num_points = random.randint(8, 15)
        points = []
        x_start = random.randint(0, w)
        y_start = 0
        
        for i in range(num_points):
            x = x_start + random.randint(-50, 50)
            y = int(h * i / num_points) + random.randint(-20, 20)
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            points.append((x, y))
        
        # Draw thin crack
        for i in range(len(points)-1):
            cv2.line(img, points[i], points[i+1], (40, 40, 40), 1)
        
        return img
    
    def add_medium_crack(self, img):
        """Add medium width crack"""
        h, w = img.shape[:2]
        
        # Generate more prominent crack
        num_points = random.randint(10, 20)
        points = []
        x_start = random.randint(w//4, 3*w//4)
        y_start = 0
        
        for i in range(num_points):
            x = x_start + random.randint(-80, 80)
            y = int(h * i / num_points) + random.randint(-30, 30)
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            points.append((x, y))
        
        # Draw thicker crack with branches
        for i in range(len(points)-1):
            thickness = random.randint(2, 4)
            cv2.line(img, points[i], points[i+1], (30, 30, 30), thickness)
            
            # Add occasional branch
            if random.random() < 0.3:
                branch_len = random.randint(20, 50)
                angle = random.uniform(-np.pi/4, np.pi/4)
                end_x = int(points[i][0] + branch_len * np.cos(angle))
                end_y = int(points[i][1] + branch_len * np.sin(angle))
                end_x = max(0, min(w-1, end_x))
                end_y = max(0, min(h-1, end_y))
                cv2.line(img, points[i], (end_x, end_y), (30, 30, 30), 2)
        
        return img
    
    def add_severe_crack(self, img):
        """Add wide, severe crack"""
        h, w = img.shape[:2]
        
        # Main crack
        num_points = random.randint(12, 25)
        points = []
        x_start = random.randint(w//3, 2*w//3)
        
        for i in range(num_points):
            x = x_start + random.randint(-100, 100)
            y = int(h * i / num_points) + random.randint(-40, 40)
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            points.append((x, y))
        
        # Draw wide crack
        for i in range(len(points)-1):
            thickness = random.randint(5, 10)
            cv2.line(img, points[i], points[i+1], (20, 20, 20), thickness)
            
            # Add spalling effect
            if random.random() < 0.4:
                spall_x = points[i][0] + random.randint(-15, 15)
                spall_y = points[i][1] + random.randint(-15, 15)
                spall_size = random.randint(10, 25)
                cv2.circle(img, (spall_x, spall_y), spall_size, (50, 50, 50), -1)
        
        # Add multiple branch cracks
        for _ in range(random.randint(2, 5)):
            start_idx = random.randint(0, len(points)-1)
            branch_len = random.randint(50, 120)
            angle = random.uniform(-np.pi/3, np.pi/3)
            end_x = int(points[start_idx][0] + branch_len * np.cos(angle))
            end_y = int(points[start_idx][1] + branch_len * np.sin(angle))
            end_x = max(0, min(w-1, end_x))
            end_y = max(0, min(h-1, end_y))
            cv2.line(img, points[start_idx], (end_x, end_y), (20, 20, 20), 
                    random.randint(3, 6))
        
        return img
    
    def add_realistic_effects(self, img):
        """Add realistic effects like lighting, shadows, noise"""
        # Random brightness adjustment
        brightness = random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
        
        # Add slight blur (camera focus variation)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Add noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def assign_split(self, index, total):
        """Assign image to train/val/test split"""
        ratio = index / total
        if ratio < 0.7:
            return 'train'
        elif ratio < 0.85:
            return 'val'
        else:
            return 'test'
    
    def print_dataset_stats(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            for cls in self.classes:
                path = os.path.join(self.output_dir, split, cls)
                count = len(os.listdir(path))
                print(f"  {cls}: {count} images")
        
        print("\n" + "="*60)
    
    def visualize_samples(self, samples_per_class=3):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(len(self.classes), samples_per_class, 
                                figsize=(15, 12))
        
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(self.output_dir, 'train', cls)
            images = os.listdir(cls_path)[:samples_per_class]
            
            for j, img_name in enumerate(images):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[i, j].imshow(img_rgb)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'{cls.replace("_", " ").title()}', 
                                        fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
        print("\n✓ Sample visualization saved as 'dataset_samples.png'")
        plt.close()
    
    def create_data_manifest(self):
        """Create JSON manifest of the dataset"""
        manifest = {
            'dataset_name': 'Concrete Crack Detection Dataset',
            'classes': self.classes,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            manifest['splits'][split] = {}
            for cls in self.classes:
                path = os.path.join(self.output_dir, split, cls)
                images = os.listdir(path)
                manifest['splits'][split][cls] = {
                    'count': len(images),
                    'images': images
                }
        
        with open(os.path.join(self.output_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=4)
        
        print("✓ Dataset manifest created: 'manifest.json'")

def main():
    print("="*70)
    print("Concrete Crack Dataset Builder")
    print("="*70)
    
    # Initialize builder
    builder = CrackDatasetBuilder(output_dir='crack_dataset')
    
    # Generate synthetic dataset
    print("\n[Option 1] Generate synthetic demonstration dataset")
    print("[Option 2] Use your own images (manual organization required)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        num_samples = int(input("Number of samples per class (default 200): ") or "200")
        builder.generate_synthetic_data(num_samples_per_class=num_samples)
        builder.visualize_samples()
        builder.create_data_manifest()
        
        print("\n" + "="*70)
        print("✓ Dataset preparation complete!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the generated samples in 'dataset_samples.png'")
        print("2. Run the training script: python train_crack_detection.py")
        print("3. For production, replace synthetic data with real crack images")
        
    else:
        print("\nManual dataset organization:")
        print("1. Place your images in the following structure:")
        print("   crack_dataset/")
        print("   ├── train/")
        print("   │   ├── no_crack/")
        print("   │   ├── hairline/")
        print("   │   ├── medium/")
        print("   │   └── severe/")
        print("   ├── val/ (same structure)")
        print("   └── test/ (same structure)")
        print("\n2. Then run: python train_crack_detection.py")

if __name__ == '__main__':
    main()