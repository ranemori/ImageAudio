import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import morphology, measure

class ImprovedSegmentationModel:
    def __init__(self, model_path=None, classes=None, device='cpu'):
        self.device = device
        self.classes = classes or ["background", "cat", "dog", "man", "woman", "rat"]
        self.model = None 
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        self.class_colors = {
            0: [0, 0, 0],        
            1: [255, 100, 100],  
            2: [100, 255, 100],    
            3: [100, 100, 255],   
            4: [255, 255, 100],  
            5: [255, 100, 255],   
        }
          
        self.coco_to_custom_mapping = {
            0: 0,    
            15: 3,   
            16: 1,   
            17: 1,   
            18: 2,   
            19: 2,    
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ Aucun modèle personnalisé trouvé, utilisation du modèle pré-entraîné")
            self.load_pretrained_model()
    
    def load_pretrained_model(self): 
        try:
            print("🔄 Chargement du modèle pré-entraîné DeepLabV3...")
             
            self.model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
            self.model = self.model.to(self.device)
            self.model.eval()
             
            self.use_coco_mapping = True
            
            print("✓ Modèle pré-entraîné chargé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle pré-entraîné: {e}")
            self.model = None
    
    def load_model(self, model_path): 
        try:
            print(f"🔄 Chargement du modèle personnalisé depuis {model_path}...")
              
            self.model = models.segmentation.deeplabv3_resnet101(weights=None)
             
            self.model.classifier[-1] = torch.nn.Conv2d(
                self.model.classifier[-1].in_channels,
                len(self.classes),
                kernel_size=1
            ) 
            if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
                self.model.aux_classifier[-1] = torch.nn.Conv2d(
                    self.model.aux_classifier[-1].in_channels,
                    len(self.classes),
                    kernel_size=1
                ) 
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
             
            self.use_coco_mapping = False
            
            print(f"✓ Modèle personnalisé chargé avec succès depuis {model_path}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle personnalisé: {e}")
            print("🔄 Fallback vers le modèle pré-entraîné...")
            self.load_pretrained_model()
    
    def preprocess_image(self, image): 
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                return None, None, None
                
            original_size = image.size
             
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            return image_tensor, original_size, image
            
        except Exception as e:
            print(f"❌ Erreur lors du préprocessing: {e}")
            return None, None, None
    
    def map_coco_to_custom(self, coco_predictions): 
        custom_predictions = np.zeros_like(coco_predictions)
          
        for coco_class, custom_class in self.coco_to_custom_mapping.items():
            mask = coco_predictions == coco_class
            custom_predictions[mask] = custom_class
          
        person_mask = coco_predictions == 15
        if np.any(person_mask):  
            person_regions = measure.label(person_mask)
            for region_id in range(1, person_regions.max() + 1):
                region_mask = person_regions == region_id 
                custom_class = 3 if np.random.random() > 0.5 else 4
                custom_predictions[region_mask] = custom_class
        
        return custom_predictions
    
    def create_precise_silhouette(self, predictions, confidence_scores): 
        try: 
            main_object_mask = np.zeros_like(predictions, dtype=bool)
            largest_area = 0
            main_class_id = 0
            
            for class_id in range(1, len(self.classes)):
                class_mask = predictions == class_id
                if np.any(class_mask): 
                    cleaned_mask = self.clean_mask_morphology(class_mask)
                     
                    labeled_mask = measure.label(cleaned_mask)
                    regions = measure.regionprops(labeled_mask)
                    
                    for region in regions:
                        if region.area > largest_area:
                            largest_area = region.area
                            main_class_id = class_id 
                            main_object_mask = labeled_mask == region.label
            
            if largest_area == 0:
                return np.zeros_like(predictions)
             
            refined_silhouette = self.refine_silhouette(
                main_object_mask, 
                confidence_scores, 
                main_class_id
            )
             
            final_mask = np.zeros_like(predictions)
            final_mask[refined_silhouette] = main_class_id
            
            return final_mask
            
        except Exception as e:
            print(f"❌ Erreur lors de la création de la silhouette: {e}")
            return predictions
    
    def clean_mask_morphology(self, mask): 
        try: 
            binary_mask = mask.astype(np.uint8)
             
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
             
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
             
            binary_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
            
            return binary_mask.astype(bool)
            
        except Exception as e:
            print(f"❌ Erreur lors du nettoyage morphologique: {e}")
            return mask.astype(bool)
    
    def refine_silhouette(self, mask, confidence_scores, class_id): 
        try: 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
             
            high_confidence_mask = confidence_scores > 0.7
            refined_region = expanded_mask.astype(bool) & high_confidence_mask
             
            final_mask = mask.copy()
             
            connection_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for _ in range(3):  
                dilated = cv2.dilate(final_mask.astype(np.uint8), connection_kernel)
                new_pixels = dilated.astype(bool) & refined_region & ~final_mask
                final_mask = final_mask | new_pixels
                if not np.any(new_pixels):
                    break
             
            final_mask = self.smooth_contours(final_mask)
            
            return final_mask
            
        except Exception as e:
            print(f"❌ Erreur lors de l'affinement: {e}")
            return mask
    
    def smooth_contours(self, mask): 
        try: 
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return mask
             
            largest_contour = max(contours, key=cv2.contourArea) 
            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True) 
            smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillPoly(smoothed_mask, [smoothed_contour], 1) 
            smoothed_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (3, 3), 0.5)
            smoothed_mask = (smoothed_mask > 0.5).astype(bool)
            
            return smoothed_mask
            
        except Exception as e:
            print(f"❌ Erreur lors du lissage: {e}")
            return mask
    
    def predict_segmentation(self, image): 
        if self.model is None:
            print("❌ Modèle non chargé")
            return None
            
        try: 
            image_tensor, original_size, original_image = self.preprocess_image(image)
            if image_tensor is None:
                return None
            
            print(f"🔍 Traitement de l'image de taille: {original_size}")
             
            with torch.no_grad():
                output = self.model(image_tensor)
                 
                if isinstance(output, dict):
                    logits = output['out']
                else:
                    logits = output
                
                print(f"📊 Forme de sortie du modèle: {logits.shape}")
                 
                probabilities = F.softmax(logits, dim=1)
                 
                predictions = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
                 
                confidence_scores = torch.max(probabilities, dim=1)[0].squeeze(0).cpu().numpy()
                
                print(f"🎯 Classes uniques détectées (avant mapping): {np.unique(predictions)}")
                  
                if hasattr(self, 'use_coco_mapping') and self.use_coco_mapping:
                    predictions = self.map_coco_to_custom(predictions)
                    print(f"🔄 Après mapping COCO - Classes uniques: {np.unique(predictions)}")
                  
                predictions_resized = cv2.resize(
                    predictions.astype(np.uint8), 
                    original_size, 
                    interpolation=cv2.INTER_NEAREST
                )
                
                confidence_resized = cv2.resize(
                    confidence_scores.astype(np.float32), 
                    original_size, 
                    interpolation=cv2.INTER_LINEAR
                )
                 
                silhouette_mask = self.create_precise_silhouette(
                    predictions_resized, 
                    confidence_resized
                )
                
                print(f"🎨 Silhouette créée - Classes finales: {np.unique(silhouette_mask)}")
                 
                return self.post_process_results(
                    silhouette_mask, 
                    confidence_resized, 
                    original_image, 
                    original_size
                )
                
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def post_process_results(self, predictions, confidence_scores, original_image, original_size): 
        try: 
            print(f"🧹 Post-traitement - Classes uniques: {np.unique(predictions)}")
              
            colored_silhouette = self.create_enhanced_colored_mask(predictions)
            overlay_image = self.create_silhouette_overlay(original_image, colored_silhouette)
            contour_image = self.draw_silhouette_contours(original_image, predictions)
            edge_image = self.create_edge_silhouette(original_image, predictions)
             
            stats = self.calculate_segmentation_stats(predictions, confidence_scores)
            
            return {
                'segmentation_map': predictions,
                'confidence_map': confidence_scores,
                'colored_mask': colored_silhouette,
                'overlay_image': overlay_image,
                'contour_image': contour_image,
                'edge_silhouette': edge_image,
                'statistics': stats,
                'detected_classes': stats['detected_classes'],
                'class_distribution': stats['class_distribution']
            }
            
        except Exception as e:
            print(f"❌ Erreur lors du post-traitement: {e}")
            return None
    
    def create_enhanced_colored_mask(self, predictions):  
        height, width = predictions.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
 
        for class_id, color in self.class_colors.items():
            if class_id < len(self.classes):
                mask = predictions == class_id
                colored_mask[mask] = color

        return Image.fromarray(colored_mask)
    
    def create_silhouette_overlay(self, original_image, colored_mask, alpha=0.7):  
        try:
            if isinstance(original_image, str):
                original_image = Image.open(original_image).convert('RGB')
                
            original_array = np.array(original_image)
            mask_array = np.array(colored_mask)
             
            if original_array.shape[:2] != mask_array.shape[:2]:
                mask_array = cv2.resize(
                    mask_array, 
                    (original_array.shape[1], original_array.shape[0])
                ) 
             
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
    
    def draw_silhouette_contours(self, original_image, predictions, thickness=3):  
        try:
            if isinstance(original_image, str):
                original_image = Image.open(original_image).convert('RGB')
                
            image_array = np.array(original_image).copy()
             
            for class_id in range(1, len(self.classes)):
                class_mask = (predictions == class_id).astype(np.uint8) * 255
                
                if np.sum(class_mask) > 0:  
                    contours, _ = cv2.findContours(
                        class_mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours and class_id in self.class_colors:
                        color = self.class_colors[class_id] 
                        color_bgr = (color[2], color[1], color[0])  
                        cv2.drawContours(image_array, contours, -1, color_bgr, thickness)
                         
                        cv2.drawContours(image_array, contours, -1, (255, 255, 255), 1)
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            print(f"❌ Erreur lors du dessin des contours: {e}")
            return original_image
    
    def create_edge_silhouette(self, original_image, predictions): 
        try:
            if isinstance(original_image, str):
                original_image = Image.open(original_image).convert('RGB')
                 
            edge_image = np.zeros_like(np.array(original_image))
            
            for class_id in range(1, len(self.classes)):
                class_mask = (predictions == class_id).astype(np.uint8) * 255
                
                if np.sum(class_mask) > 0: 
                    edges = cv2.Canny(class_mask, 50, 150)
                     
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    edges = cv2.dilate(edges, kernel, iterations=1)
                     
                    if class_id in self.class_colors:
                        color = self.class_colors[class_id]
                        edge_image[edges > 0] = color
            
            return Image.fromarray(edge_image)
            
        except Exception as e:
            print(f"❌ Erreur lors de la création des bords: {e}")
            return original_image
    
    def calculate_segmentation_stats(self, predictions, confidence_scores):  
        try:
            unique_classes = np.unique(predictions)
            detected_classes = []
            class_distribution = {}
            total_pixels = predictions.size
            
            for class_id in unique_classes:
                if class_id < len(self.classes) and class_id > 0:  
                    class_name = self.classes[class_id]
                    class_mask = predictions == class_id
                    pixel_count = np.sum(class_mask)
                    percentage = (pixel_count / total_pixels) * 100
                     
                    if np.any(class_mask):
                        avg_confidence = np.mean(confidence_scores[class_mask])
                    else:
                        avg_confidence = 0.0 
                    if percentage > 1.0:  
                        detected_classes.append(class_name)
                        class_distribution[class_name] = {
                            'pixels': int(pixel_count),
                            'percentage': round(percentage, 2),
                            'avg_confidence': round(avg_confidence, 3),
                            'color': self.class_colors.get(class_id, [128, 128, 128])
                        }
            
            return {
                'detected_classes': detected_classes,
                'class_distribution': class_distribution,
                'total_pixels': total_pixels,
                'num_classes_detected': len(detected_classes)
            }
            
        except Exception as e:
            print(f"❌ Erreur lors du calcul des statistiques: {e}")
            return {
                'detected_classes': [],
                'class_distribution': {},
                'total_pixels': 0,
                'num_classes_detected': 0
            }
    
    def visualize_results(self, results, save_path=None):  
        if results is None:
            print("❌ Aucun résultat à visualiser")
            return
            
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
            axes[0, 0].imshow(results['colored_mask'])
            axes[0, 0].set_title('Silhouette Colorée', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off') 
            axes[0, 1].imshow(results['overlay_image'])
            axes[0, 1].set_title('Superposition sur Image Originale', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off') 
            axes[0, 2].imshow(results['contour_image'])
            axes[0, 2].set_title('Contours de la Silhouette', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off') 
            axes[1, 0].imshow(results['edge_silhouette'])
            axes[1, 0].set_title('Silhouette - Bords Seulement', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off') 
            axes[1, 1].imshow(results['confidence_map'], cmap='viridis')
            axes[1, 1].set_title('Carte de Confiance', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off') 
            if results['statistics']['class_distribution']:
                class_names = list(results['statistics']['class_distribution'].keys())
                percentages = [results['statistics']['class_distribution'][name]['percentage'] 
                             for name in class_names]
                colors = [np.array(results['statistics']['class_distribution'][name]['color'])/255.0 
                         for name in class_names]
                
                bars = axes[1, 2].bar(class_names, percentages, color=colors)
                axes[1, 2].set_title('Distribution des Classes (%)', fontsize=14, fontweight='bold')
                axes[1, 2].set_ylabel('Pourcentage')
                axes[1, 2].tick_params(axis='x', rotation=45)
                 
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{pct:.1f}%', ha='center', va='bottom')
            else:
                axes[1, 2].text(0.5, 0.5, 'Aucune classe détectée', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Distribution des Classes')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Résultats sauvegardés dans {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Erreur lors de la visualisation: {e}")

 
def predict_with_improved_model(model_path, image_path, classes=None, device='cpu'): 
    if classes is None:
        classes = ["background", "cat", "dog", "man", "woman", "rat"]
    
    print(f"🔍 Chargement du modèle depuis: {model_path}")
    print(f"🖼️ Traitement de l'image: {image_path}")
    
    model = ImprovedSegmentationModel(model_path, classes, device)
    results = model.predict_segmentation(image_path)
    
    if results:
        print("\n📊 Résultats de la segmentation:")
        print(f"Classes détectées: {results['detected_classes']}")
        print("\n📈 Distribution des classes:")
        for class_name, info in results['class_distribution'].items():
            print(f"  {class_name}: {info['percentage']:.1f}% "
                  f"(confiance: {info['avg_confidence']:.3f})")
         
        model.visualize_results(results)
        
    return results
 
if __name__ == "__main__":
    print("🧪 Test du modèle de segmentation avec silhouettes...")
      
    test_image = Image.new('RGB', (256, 256), color='white')
 
    model = ImprovedSegmentationModel()
    if model.model is not None:
        results = model.predict_segmentation(test_image)
        if results:
            print("✓ Test réussi - Le modèle fonctionne")
            print(f"Classes détectées: {results['detected_classes']}")
        else:
            print("❌ Test échoué - Pas de résultats")
    else:
        print("❌ Test échoué - Modèle non chargé")