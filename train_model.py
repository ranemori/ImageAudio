import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter
import argparse
import json

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, classes=None, transform=None, mask_transform=None, mode='classification'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode
        self.classes = classes or ["background", "cat", "dog", "man", "woman", "rat"] 
        self.image_paths = []
        self.mask_paths = [] 
        
        if mode == 'segmentation' and masks_dir:
            self._collect_segmentation_pairs()
            
    def _collect_segmentation_pairs(self):  
        for category in self.classes[1:]:   
            category_dir = os.path.join(self.images_dir, category)
            if os.path.exists(category_dir):
                for img_file in os.listdir(category_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_path = os.path.join(category_dir, img_file)  
                        mask_name = img_file.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.webp', '.png')
                        mask_path = os.path.join(self.masks_dir, mask_name) 
                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                            
    def __len__(self):
        return len(self.image_paths) 
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.mode == 'segmentation':
            mask = Image.open(self.mask_paths[idx]).convert('L') 
            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)
                mask = torch.from_numpy(np.array(mask, dtype=np.int64))
            else: 
                mask = mask.resize((512, 512), Image.NEAREST)  
                mask = torch.from_numpy(np.array(mask, dtype=np.int64))
                
            return image, mask
        else: 
            if self.transform:
                image = self.transform(image)
            return image, 0  

class ModelTrainer:
    def __init__(self, data_dir="./dataset", batch_size=16, num_epochs=25, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["background", "cat", "dog", "man", "woman", "rat"] 
        print(f"  Utilisation de {self.device} pour l'entraînement")
        print(f"  Classes définies: {self.classes}") 
         
        self.classification_train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 
        
        self.classification_val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 
        
        self.segmentation_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
         
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST)
        ])
        
    def verify_dataset_structure(self, mode='classification'): 
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Le dossier '{self.data_dir}' n'existe pas.") 
            
        if mode == 'classification': 
            train_images_dir = os.path.join(self.data_dir, "train", "images")
            val_images_dir = os.path.join(self.data_dir, "val", "images") 
            
            print(f"  Vérification de la structure pour classification...")
            print(f"  Structure attendue: {self.data_dir}/[train|val]/images/class_name/") 
            
            if not os.path.exists(train_images_dir):
                print(f"❌ Dossier d'entraînement manquant: {train_images_dir}")
                return False
            
            if not os.path.exists(val_images_dir):
                print(f"❌ Dossier de validation manquant: {val_images_dir}")
                return False 
                
            for class_name in self.classes[1:]:  
                train_class_dir = os.path.join(train_images_dir, class_name)
                val_class_dir = os.path.join(val_images_dir, class_name) 
                
                if os.path.exists(train_class_dir):
                    count = len([f for f in os.listdir(train_class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"✅ {class_name} (train): {count} images")
                else:
                    print(f"❌ Dossier manquant: {train_class_dir}") 
                    
                if os.path.exists(val_class_dir):
                    count = len([f for f in os.listdir(val_class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"✅ {class_name} (val): {count} images")
                else:
                    print(f"❌ Dossier manquant: {val_class_dir}") 
                    
        elif mode == 'segmentation':
            return self._verify_segmentation_structure()
            
        return True
    
    def _verify_segmentation_structure(self): 
        train_images_dir = os.path.join(self.data_dir, "train", "images")
        train_masks_dir = os.path.join(self.data_dir, "train", "masks")
        val_images_dir = os.path.join(self.data_dir, "val", "images")
        val_masks_dir = os.path.join(self.data_dir, "val", "masks") 
        
        print(f"  Vérification de la structure pour segmentation...")
        print(f"  Structure attendue: {self.data_dir}/[train|val]/[images|masks]/")
        
        if not os.path.exists(train_images_dir):
            print(f"❌ Dossier d'images d'entraînement manquant: {train_images_dir}")
            return False 
        if not os.path.exists(train_masks_dir):
            print(f"❌ Dossier de masques d'entraînement manquant: {train_masks_dir}")
            return False 
        if not os.path.exists(val_images_dir):
            print(f"❌ Dossier d'images de validation manquant: {val_images_dir}")
            return False 
        if not os.path.exists(val_masks_dir):
            print(f"❌ Dossier de masques de validation manquant: {val_masks_dir}")
            return False 
            
        train_images_count = 0
        train_masks_count = 0
        
        for category in self.classes[1:]: 
            category_img_dir = os.path.join(train_images_dir, category)
            if os.path.exists(category_img_dir):
                category_images = [f for f in os.listdir(category_img_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                train_images_count += len(category_images)
                print(f"✅ {category} (train images): {len(category_images)}") 
                
        if os.path.exists(train_masks_dir):
            train_masks = [f for f in os.listdir(train_masks_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            train_masks_count = len(train_masks) 
            
        val_images_count = 0
        val_masks_count = 0 
        
        for category in self.classes[1:]:  
            category_img_dir = os.path.join(val_images_dir, category)
            if os.path.exists(category_img_dir):
                category_images = [f for f in os.listdir(category_img_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                val_images_count += len(category_images)
                print(f"✅ {category} (val images): {len(category_images)}") 
                
        if os.path.exists(val_masks_dir):
            val_masks = [f for f in os.listdir(val_masks_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            val_masks_count = len(val_masks) 
            
        print(f"  Images d'entraînement: {train_images_count}")
        print(f"  Masques d'entraînement: {train_masks_count}")
        print(f"  Images de validation: {val_images_count}")
        print(f"  Masques de validation: {val_masks_count}") 
        
        return train_images_count > 0 and train_masks_count > 0 and val_images_count > 0 and val_masks_count > 0
        
    def load_classification_datasets(self): 
        try: 
            train_images_dir = os.path.join(self.data_dir, "train", "images")
            val_images_dir = os.path.join(self.data_dir, "val", "images") 
            
            print(f"  Chargement depuis:")
            print(f"  - Train: {train_images_dir}")
            print(f"  - Val: {val_images_dir}") 
            
            train_dataset = datasets.ImageFolder(
                train_images_dir, 
                transform=self.classification_train_transform
            )
            val_dataset = datasets.ImageFolder(
                val_images_dir, 
                transform=self.classification_val_transform
            ) 
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            ) 
            
            print(f"  Classes détectées: {train_dataset.classes}")
            print(f"  Images d'entraînement: {len(train_dataset)}")
            print(f"  Images de validation: {len(val_dataset)}") 
            
            train_counts = Counter([train_dataset.classes[label] for _, label in train_dataset])
            print("  Distribution des classes (train):")
            for cls, count in train_counts.items():
                print(f"  - {cls}: {count} images") 
                
            return len(train_dataset.classes) 
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des datasets de classification: {e}")
            raise 
            
    def load_segmentation_datasets(self): 
        try:
            train_images_dir = os.path.join(self.data_dir, "train", "images")
            train_masks_dir = os.path.join(self.data_dir, "train", "masks")
            val_images_dir = os.path.join(self.data_dir, "val", "images")
            val_masks_dir = os.path.join(self.data_dir, "val", "masks") 
            
            if not os.path.exists(train_images_dir) or not os.path.exists(train_masks_dir):
                raise FileNotFoundError("Dossiers d'entraînement manquants.")
            if not os.path.exists(val_images_dir) or not os.path.exists(val_masks_dir):
                raise FileNotFoundError("Dossiers de validation manquants.") 
                
            train_dataset = CustomDataset(
                images_dir=train_images_dir,
                masks_dir=train_masks_dir,
                classes=self.classes,
                transform=self.segmentation_transform,
                mask_transform=self.mask_transform,
                mode='segmentation'
            )
            val_dataset = CustomDataset(
                images_dir=val_images_dir,
                masks_dir=val_masks_dir,
                classes=self.classes,
                transform=self.segmentation_transform,
                mask_transform=self.mask_transform,
                mode='segmentation'
            ) 
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            ) 
            
            print(f"  Images d'entraînement: {len(train_dataset)}")
            print(f"  Images de validation: {len(val_dataset)}")
            print(f"  Nombre de classes: {len(self.classes)}")
            return len(self.classes)
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des datasets de segmentation: {e}")
            raise
            
    def create_classification_model(self, num_classes): 
        print(f"  Création du modèle de classification pour {num_classes} classes...") 
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')  
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model.to(self.device)
        
    def create_segmentation_model(self, num_classes): 
        print(f"  Création du modèle de segmentation pour {num_classes} classes...")
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='COCO_WITH_VOC_LABELS_V1') 
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1) 
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, num_classes, kernel_size=1)
        return model.to(self.device)
        
    def train_classification_model(self): 
        print("\n" + "="*60)
        print("  ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION")
        print("="*60) 
        
        if not self.verify_dataset_structure('classification'):
            print("❌ Structure de dataset invalide pour classification")
            return None 
            
        num_classes = self.load_classification_datasets()
        model = self.create_classification_model(num_classes) 
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 
        
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = [] 
        
        for epoch in range(self.num_epochs): 
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0 
            
            train_bar = tqdm(self.train_loader, desc=f"  Epoch {epoch+1}/{self.num_epochs}") 
            
            for images, labels in train_bar:
                images, labels = images.to(self.device), labels.to(self.device) 
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step() 
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0) 
                
                current_acc = correct / total
                train_bar.set_postfix({
                    'Loss': f'{running_loss/(train_bar.n+1):.4f}', 
                    'Acc': f'{current_acc:.4f}'
                }) 
                
            val_loss, val_acc = self.validate_classification_model(model, criterion)
            scheduler.step() 
            
            train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total 
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc) 
            
            print(f"\n  Epoch {epoch+1}/{self.num_epochs}:")
            print(f"    Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"    Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}") 
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = "mobilenet_finetuned.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'classes': [cls for cls in os.listdir(os.path.join(self.data_dir, "train", "images")) 
                           if os.path.isdir(os.path.join(self.data_dir, "train", "images", cls))]
                }, model_save_path)
                print(f"    Meilleur modèle sauvegardé: {model_save_path} (Acc: {val_acc:.4f})") 
                
            print("-" * 50) 
            
        print(f"\n  Entraînement de classification terminé!")
        print(f"  Meilleure précision de validation: {best_val_acc:.4f}") 
        
        return model
        
    def train_segmentation_model(self):
     print("=" * 60)
     print("🎯 ENTRAÎNEMENT DU MODÈLE DE SEGMENTATION")
     print("=" * 60)

     if not self.verify_dataset_structure('segmentation'):
        print("❌ Structure de dataset invalide pour segmentation")
        return None

     num_classes = self.load_segmentation_datasets()
     model = self.create_segmentation_model(num_classes)
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

     best_val_loss = float('inf')
     train_losses = []
     val_losses = []

     for epoch in range(self.num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for images, masks in train_bar:
            images, masks = images.to(self.device), masks.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{running_loss / (train_bar.n + 1):.4f}'})
 
        model.eval()
        val_loss = 0.0
        iou_scores = []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)['out']

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                masks = masks.cpu().numpy()

                for pred, mask in zip(predictions, masks):
                    iou = calculate_iou(pred, mask)
                    iou_scores.append(iou)

        mean_iou = np.mean(iou_scores)
        print(f"📊 Epoch {epoch+1}/{self.num_epochs}:")
        print(f" 🚂 Train Loss: {running_loss / len(self.train_loader):.4f}")
        print(f" ✅ Val Loss: {val_loss / len(self.val_loader):.4f}, Mean IoU: {mean_iou:.4f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "segmentation_model.pth")
            print("💾 Meilleur modèle sauvegardé.")

     print("🎉 Entraînement terminé.")
        
    def validate_classification_model(self, model, criterion): 
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0 
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device) 
                
                outputs = model(images)
                loss = criterion(outputs, labels) 
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0) 
                
        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy 
        
    def validate_segmentation_model(self, model, criterion): 
        model.eval()
        running_loss = 0.0 
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device) 
                
                outputs = model(images) 
                
                if isinstance(outputs, dict):
                    main_output = outputs['out']
                    aux_output = outputs.get('aux', None)
                else:
                    main_output = outputs
                    aux_output = None 
                    
                main_loss = criterion(main_output, masks) 
                
                if aux_output is not None:
                    aux_loss = criterion(aux_output, masks)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    loss = main_loss 
                    
                running_loss += loss.item() 
                
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss 
        
    def save_training_info(self, model_type, best_metric): 
        info = {
            'model_type': model_type,
            'classes': self.classes,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
            'best_metric': best_metric
        } 
        
        with open(f'{model_type}_training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"  Informations d'entraînement sauvegardées: {model_type}_training_info.json") 

def main():
    parser = argparse.ArgumentParser(description='Entraînement des modèles personnalisés')
    parser.add_argument('--model', choices=['classification', 'segmentation', 'both'], 
                       default='classification', help='Type de modèle à entraîner')
    parser.add_argument('--epochs', type=int, default=25, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille du batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Taux d\'apprentissage')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Répertoire des données') 
    
    args = parser.parse_args() 
    
    print("  DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("="*60)
    print(f"  Répertoire de données: {args.data_dir}")
    print(f"  Modèle(s) à entraîner: {args.model}")
    print(f"  Nombre d'époques: {args.epochs}")
    print(f"  Taille de batch: {args.batch_size}")
    print(f"  Taux d'apprentissage: {args.learning_rate}")
    print("="*60) 
    
    try:
        trainer = ModelTrainer(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        ) 
        
        trained_models = [] 
        
        if args.model in ['classification', 'both']:
            print("\n  Entraînement du modèle de classification...")
            model = trainer.train_classification_model()
            if model is not None:
                trained_models.append('classification')
                trainer.save_training_info('classification', 'accuracy')
                
        if args.model in ['segmentation', 'both']:
            print("\n  Entraînement du modèle de segmentation...")
            model = trainer.train_segmentation_model()
            if model is not None:
                trained_models.append('segmentation')
                trainer.save_training_info('segmentation', 'loss')
                
        print("\n" + "="*60)
        print("  ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        print("  Modèles sauvegardés:")
        
        if "classification" in trained_models and os.path.exists("mobilenet_finetuned.pth"):
            print("✅ mobilenet_finetuned.pth (classification)")
        
        if "segmentation" in trained_models and os.path.exists("segmentation_model.pth"):
            print("✅ segmentation_model.pth (segmentation)")
        
        if not trained_models:
            print("❌ Aucun modèle n'a été entraîné avec succès")
        else:
            print(f"\n  {len(trained_models)} modèle(s) entraîné(s) avec succès!")
            print("\n  Pour utiliser les modèles:")
            print("   - Classification: utilisez model.py avec mobilenet_finetuned.pth")
            print("   - Segmentation: utilisez segmentation_model.py avec segmentation_model.pth")
            
    except KeyboardInterrupt:
        print("\n  Entraînement interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()