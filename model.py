import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2

class CustomModel:
    def __init__(self, model_path, classes, device, model_type='classification'):
        self.device = device
        self.classes = classes
        self.model_type = model_type
        self.model = None
         
        if model_type == 'segmentation':
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),   
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            if self.model_type == 'classification':
                self.model = models.mobilenet_v2(weights=None)
                self.model.classifier[1] = torch.nn.Linear(
                    self.model.classifier[1].in_features, 
                    len(self.classes)
                )
            elif self.model_type == 'segmentation':
                self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None) 
                num_classes = len(self.classes) + 1  
                self.model.classifier[-1] = torch.nn.Conv2d(
                    self.model.classifier[-1].in_channels,
                    num_classes,
                    kernel_size=1
                ) 
                if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
                    self.model.aux_classifier[-1] = torch.nn.Conv2d(
                        self.model.aux_classifier[-1].in_channels,
                        num_classes,
                        kernel_size=1
                    )
             
            if model_path and torch.cuda.is_available() or model_path:
                checkpoint = torch.load(model_path, map_location=self.device)
                 
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                 
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Modèle {self.model_type} chargé avec succès depuis {model_path}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle {self.model_type}: {e}") 
            if self.model_type == 'segmentation':
                print("🔄 Chargement des poids pré-entraînés ImageNet...")
                self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
                num_classes = len(self.classes) + 1
                self.model.classifier[-1] = torch.nn.Conv2d(
                    self.model.classifier[-1].in_channels,
                    num_classes,
                    kernel_size=1
                )
                if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
                    self.model.aux_classifier[-1] = torch.nn.Conv2d(
                        self.model.aux_classifier[-1].in_channels,
                        num_classes,
                        kernel_size=1
                    )
                self.model = self.model.to(self.device)
                self.model.eval()
            else:
                self.model = None
    
    def predict(self, image):
        if self.model is None:
            return None
            
        try: 
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                return None
            
            original_size = image.size
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.model_type == 'classification':
                    return self.predict_classification(image_tensor)
                elif self.model_type == 'segmentation':
                    return self.predict_segmentation(image_tensor, original_size, image)
                    
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def predict_classification(self, image_tensor):
        try:
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            if predicted_idx.item() < len(self.classes):
                class_name = self.classes[predicted_idx.item()]
            else:
                class_name = "unknown"
            
            confidence_score = confidence.item()
            
            return {
                'class': class_name,
                'confidence': confidence_score,
                'probabilities': probabilities.cpu().numpy()[0]
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de la classification: {e}")
            return None
    
    def predict_segmentation(self, image_tensor, original_size, original_image):
        try:
            output = self.model(image_tensor)
            
            if isinstance(output, dict):
                segmentation_output = output['out']
            else:
                segmentation_output = output
             
            probabilities = F.softmax(segmentation_output, dim=1)
             
            predictions = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
             
            confidence_scores = torch.max(probabilities, dim=1)[0].squeeze(0).cpu().numpy()
              
            predictions = cv2.resize(
                predictions.astype(np.uint8), 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            confidence_scores = cv2.resize(
                confidence_scores, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
              
            confidence_threshold = 0.7  
            filtered_predictions = predictions.copy()
            low_confidence_mask = confidence_scores < confidence_threshold
            filtered_predictions[low_confidence_mask] = 0
             
            filtered_predictions = self.clean_segmentation_mask(filtered_predictions)
             
            filtered_predictions = self.keep_dominant_object(filtered_predictions, confidence_scores)
             
            colored_mask = self.create_colored_segmentation_mask(filtered_predictions)
             
            overlay_image = self.create_segmentation_overlay(original_image, colored_mask)
             
            contour_image = self.draw_segmentation_contours(original_image, filtered_predictions)
             
            unique_classes = np.unique(filtered_predictions)
            detected_classes = []
            class_distribution = {}
            total_pixels = filtered_predictions.size
            
            for class_id in unique_classes:
                if class_id > 0 and class_id <= len(self.classes): 
                    class_name = self.classes[class_id - 1]  
                    class_mask = filtered_predictions == class_id
                    pixel_count = np.sum(class_mask)
                    percentage = (pixel_count / total_pixels) * 100
                    avg_confidence = np.mean(confidence_scores[class_mask])
                     
                    if percentage > 2.0:  
                        detected_classes.append(class_name)
                        class_distribution[class_name] = {
                            'pixels': int(pixel_count),
                            'percentage': round(percentage, 2),
                            'avg_confidence': round(avg_confidence, 3)
                        }
            
            return {
                'segmentation_map': filtered_predictions,
                'confidence_map': confidence_scores,
                'colored_mask': colored_mask,
                'overlay_image': overlay_image,
                'contour_image': contour_image,
                'detected_classes': detected_classes,
                'class_distribution': class_distribution
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de la segmentation: {e}")
            return None
    
    def clean_segmentation_mask(self, predictions): 
        try:
            cleaned = predictions.copy()
             
            for class_id in range(1, len(self.classes) + 1):
                class_mask = (predictions == class_id).astype(np.uint8)
                
                if np.sum(class_mask) > 0: 
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
                    class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
                     
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                     
                    min_area = max(500, predictions.size * 0.001)   
                     
                    clean_mask = np.zeros_like(class_mask)
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area >= min_area:
                            cv2.fillPoly(clean_mask, [contour], 1)
                     
                    cleaned[predictions == class_id] = 0
                    cleaned[clean_mask == 1] = class_id
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Erreur lors du nettoyage: {e}")
            return predictions
    
    def keep_dominant_object(self, predictions, confidence_scores): 
        try: 
            class_scores = {}
            
            for class_id in range(1, len(self.classes) + 1):
                class_mask = predictions == class_id
                if np.any(class_mask): 
                    pixel_count = np.sum(class_mask)
                    avg_confidence = np.mean(confidence_scores[class_mask])
                    area_ratio = pixel_count / predictions.size
                     
                    composite_score = area_ratio * avg_confidence
                    class_scores[class_id] = {
                        'score': composite_score,
                        'pixels': pixel_count,
                        'confidence': avg_confidence,
                        'area_ratio': area_ratio
                    }
            
            if not class_scores:
                return predictions 
            if len(class_scores) > 1: 
                sorted_classes = sorted(class_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                 
                dominant_class, dominant_info = sorted_classes[0] 
                if dominant_info['area_ratio'] < 0.05: 
                    return np.zeros_like(predictions)
                 
                result = np.zeros_like(predictions)
                result[predictions == dominant_class] = dominant_class
                
                print(f"🎯 Objet dominant détecté: classe {dominant_class} "
                      f"(score: {dominant_info['score']:.3f}, "
                      f"aire: {dominant_info['area_ratio']*100:.1f}%)")
                
                return result
            
            return predictions
            
        except Exception as e:
            print(f"❌ Erreur lors de la sélection de l'objet dominant: {e}")
            return predictions
    
    def create_colored_segmentation_mask(self, predictions):
        colors = [
            [0, 0, 0],           
            [255, 0, 0],        
            [0, 255, 0],        
            [0, 0, 255],      
            [255, 255, 0],       
            [255, 0, 255],       
            [0, 255, 255],      
            [255, 128, 0],       
            [128, 255, 0],       
            [255, 0, 128],      
        ]
          
        while len(colors) <= len(self.classes):
            colors.append([
                np.random.randint(100, 255),
                np.random.randint(100, 255),
                np.random.randint(100, 255)
            ])
         
        colored_mask = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(len(colors)):
            if class_id <= len(self.classes):  
                mask = predictions == class_id
                colored_mask[mask] = colors[class_id]
        
        return Image.fromarray(colored_mask)
    
    def create_segmentation_overlay(self, original_image, colored_mask, alpha=0.5): 
        try: 
            original_array = np.array(original_image.convert('RGB'))
            mask_array = np.array(colored_mask)
             
            if original_array.shape[:2] != mask_array.shape[:2]:
                mask_array = cv2.resize(mask_array, (original_array.shape[1], original_array.shape[0]))
             
            background_mask = np.all(mask_array == [0, 0, 0], axis=2)
            alpha_map = np.full(background_mask.shape, alpha)
            alpha_map[background_mask] = 0.0  
             
            overlay = original_array.copy().astype(float)
            mask_array = mask_array.astype(float)
            
            for i in range(3):   
                overlay[:, :, i] = (1 - alpha_map) * original_array[:, :, i] + \
                                  alpha_map * mask_array[:, :, i]
            
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            return Image.fromarray(overlay)
            
        except Exception as e:
            print(f"❌ Erreur lors de la création de la superposition: {e}")
            return original_image
    
    def draw_segmentation_contours(self, original_image, predictions, thickness=2): 
        try:
            image_array = np.array(original_image.convert('RGB')).copy()
             
            colors = [
                (255, 0, 0),      
                (0, 255, 0),       
                (255, 0, 0),       
                (0, 255, 255),    
                (255, 0, 255),    
                (255, 255, 0),   
                (0, 128, 255),  
                (0, 255, 128),    
                (128, 0, 255),    
            ]
            
            for class_id in range(1, len(self.classes) + 1):   
                class_mask = (predictions == class_id).astype(np.uint8) * 255
                
                if np.sum(class_mask) > 0:   
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
                    class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
                     
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours and (class_id - 1) < len(colors):
                        color = colors[class_id - 1]
                        cv2.drawContours(image_array, contours, -1, color, thickness)
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            print(f"❌ Erreur lors du dessin des contours: {e}")
            return original_image