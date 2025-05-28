import os
import io
from io import BytesIO
import numpy as np
import wave
from gtts import gTTS
import tempfile
import shutil

class SoundManager:
    def __init__(self, sounds_dir='sounds'):
        self.sounds_dir = sounds_dir
        self.animal_sounds = {}
        self.human_voices = {}
        self.load_sounds()
    
    def load_sounds(self):
        try:
            os.makedirs(self.sounds_dir, exist_ok=True)

            sound_files = {
                'cat': 'cat_meow.mp3',
                'dog': 'dog_bark.mp3',
                'rat': 'rat_squeak.mp3',
                'man': 'man_hello.mp3',
                'woman': 'woman_hello.mp3'
            }
            
            # Charger fichier son
            for class_name, filename in sound_files.items():
                filepath = os.path.join(self.sounds_dir, filename)

                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'rb') as f:
                            if class_name in ['cat', 'dog', 'rat']:
                                self.animal_sounds[class_name] = BytesIO(f.read())
                            else:
                                self.human_voices[class_name] = BytesIO(f.read())
                        print(f"✓ Son de {class_name} chargé depuis {filepath}")
                    except Exception as e:
                        print(f"Erreur lors du chargement du son pour {class_name}: {e}")
                else:
                    print(f"⚠ Fichier son manquant: {filepath}")
                    
        except Exception as e:
            print(f"Erreur générale lors du chargement des sons: {e}")

    def get_sound_for_class(self, class_name):
        try:
            class_name = class_name.lower().strip()
            
            class_mapping = {
                'cat': 'cat',
                'cats': 'cat',
                'kitten': 'cat',
                'feline': 'cat',
                'dog': 'dog',
                'dogs': 'dog',
                'puppy': 'dog',
                'canine': 'dog',
                'rat': 'rat',
                'rats': 'rat',
                'mouse': 'rat',
                'rodent': 'rat',
                'man': 'man',
                'men': 'man',
                'boy': 'man',
                'male': 'man',
                'woman': 'woman',
                'women': 'woman',
                'girl': 'woman',
                'female': 'woman'
            }
            
            mapped_class = None
            for key, value in class_mapping.items():
                if key in class_name or class_name in key:
                    mapped_class = value
                    break
            
            if not mapped_class:
                return None
            
            if mapped_class in self.animal_sounds:
                return self.animal_sounds[mapped_class]
            elif mapped_class in self.human_voices:
                return self.human_voices[mapped_class]
            
            return None
            
        except Exception as e:
            print(f"Erreur dans get_sound_for_class: {e}")
            return None