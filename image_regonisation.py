import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from gtts import gTTS
from io import BytesIO
import os
import numpy as np

class ImageRecognitionToSpeech:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["cat", "dog", "man", "rat", "woman"]
        self.model = None
        
        # Transformations pour l'image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if model_path and os.path.exists(model_path):
            self.load_custom_model(model_path)
        else:
            print(f"Modèle personnalisé non trouvé: {model_path}")
            self.model = None
    
    def load_custom_model(self, model_path):
        try:
            # Créer le modèle 
            self.model = models.mobilenet_v2(weights=None)

            self.model.classifier[1] = torch.nn.Linear(
                self.model.classifier[1].in_features, 
                len(self.classes)
            )
            
            # Charger les poids du modèle
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Modèle personnalisé chargé depuis {model_path}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle personnalisé: {e}")
            self.model = None
    
    def recognize_image(self, image):
        if self.model is None:
            print("❌ Aucun modèle disponible")
            return "unknown", 0.0
        
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                print("❌ Format d'image non supporté")
                return "unknown", 0.0

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            if predicted_idx.item() < len(self.classes):
                class_name = self.classes[predicted_idx.item()]
            else:
                class_name = "unknown"
            
            confidence_score = confidence.item()
            
            print(f"✓ Prédiction: {class_name} (confiance: {confidence_score:.4f})")
            return class_name, confidence_score
            
        except Exception as e:
            print(f"❌ Erreur lors de la reconnaissance d'image: {e}")
            return "unknown", 0.0
    
    def text_to_speech(self, text, lang='en', voice_type=None):
        try:
            tld = None
            if voice_type == 'man':
                tld = 'com'  
            elif voice_type == 'woman':
                tld = 'co.uk'  

            if tld:
                tts = gTTS(text=text, lang=lang, tld=tld)
            else:
                tts = gTTS(text=text, lang=lang)
            
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération TTS: {e}")
            return BytesIO()
    
    def predict_image_with_audio(self, image, lang='en'):
        try:
            # Reconnaissance de l'image
            class_name, confidence_score = self.recognize_image(image)
            
            if class_name == "unknown":
                return {
                    'class': 'unknown',
                    'confidence': 0.0,
                    'text': 'Unable to recognize the object. Please train the custom model first.',
                    'audio': self.text_to_speech('Unable to recognize the object. Please train the custom model first.')
                }
       
            text = f"The image contains a {class_name} with {int(confidence_score*100)} percent confidence."
            
            # Déterminer le type de voix
            voice_type = None
            if class_name in ['man']:
                voice_type = 'man'
            elif class_name in ['woman']:
                voice_type = 'woman'
            
            # Générer l'audio avec la voix appropriée
            audio_buffer = self.text_to_speech(text, lang, voice_type)
            
            return {
                'class': class_name,
                'confidence': confidence_score,
                'text': text,
                'audio': audio_buffer
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction avec audio: {e}")
            return {
                'class': 'error',
                'confidence': 0.0,
                'text': 'Error occurred during prediction.',
                'audio': self.text_to_speech('Error occurred during prediction.')
            }
    
    def get_model_info(self):
        if self.model is None:
            return {
                'status': 'No model loaded',
                'classes': self.classes,
                'device': str(self.device),
                'ready': False
            }
        else:
            return {
                'status': 'Custom model loaded',
                'classes': self.classes,
                'device': str(self.device),
                'ready': True,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
    
    def detect_objects_with_boxes(self, image):
    if not hasattr(self, "detection_model"):
        st.error("Modèle de détection non disponible")
        return None
    try:
        predictions = self.detection_model.predict(image)
        image_array = np.array(image)
        for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
            if score > 0.5:  # Seuil de confiance
                box = box.detach().cpu().numpy().astype(int)
                cv2.rectangle(image_array, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(image_array, f"{label}: {score:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        annotated_image = Image.fromarray(image_array)
        return {
            "annotated_image": annotated_image,
            "labels": predictions["labels"],
            "scores": predictions["scores"]
        }
    except Exception as e:
        st.error(f"Erreur lors de la détection: {e}")
        return None
    
    def is_ready(self):
        return self.model is not None