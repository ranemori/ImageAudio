import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
from sound_manager import SoundManager
from model import CustomModel
from train_model import ModelTrainer
import tempfile
import base64
from io import BytesIO
import cv2

st.set_page_config(
    page_title="IA Vision & Audio",
    page_icon="icon.png",
    layout="wide"
)

st.markdown("""
    <style>
    .st-emotion-cache-iyz50i {
        padding: 1.25rem 1.75rem;
        color: rgb(192 225 255);
        background-color: violet;
        border: 2px solid rgb(192 225 255);
    }
    .st-emotion-cache-10c9vv9 {
        font-size: 2rem;
    }
    .st-emotion-cache-102y9h7 {
        font-size: 2.7rem;
        margin-bottom: 5rem;
        color: rgb(247 4 215);
    }
    .st-emotion-cache-qsto9u {
        min-height: 4.5rem;
        background-color: rgb(192 225 255);
        border: 1px solid rgba(255, 75, 75, 0.5);
        color: rgb(0 6 41);
        margin-bottom: 40px;
    }
    .st-emotion-cache-p7i6r9 {
        font-size: 2rem; 
    }
    .st-emotion-cache-1fmytai {
        color: violet;
    }
    .st-emotion-cache-16tyu1 {
        font-size: 1.5rem;
        color: rgb(39, 110, 241);
    }
    .st-emotion-cache-1gulkj5 {
        background-color: rgb(185 222 255);
        color: rgb(247 4 215);
        width: 100%;
        height: 6.5rem;
    }
    .st-emotion-cache-6rlrad {
        width: 5.3rem;
        height: 5.3rem;
    }
    .st-emotion-cache-j7qwjs {
        margin-left: 40px;
        font-size: 20px;
    }
    .st-emotion-cache-ocsh0s {
        min-height: 4.5rem;
    }
    .st-emotion-cache-nwtri {
        color: rgb(255, 255, 255);
        margin-right: 3rem;
        margin-left: 2rem;
    } 
    .st-emotion-cache-15ibi94 {
        width: 400px;
    } 
    .st-emotion-cache-1binpeo {
        height: 3.5rem;
        width: 100%;
        margin: 2px;
        background-color: rgb(192 225 255);
    } 
    </style>
""", unsafe_allow_html=True)

class ImageProcessingApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["cat", "dog", "man", "woman", "rat"]
        self.setup_models()
        self.setup_audio()

    def setup_models(self):
        try:
            classification_model_path = "mobilenet_finetuned.pth"
            segmentation_model_path = "segmentation_model.pth"
            if os.path.exists(classification_model_path):
                self.classification_model = CustomModel(
                    model_path=classification_model_path,
                    classes=self.classes,
                    device=self.device,
                    model_type="classification"
                )
            else:
                st.warning("⚠ Classification model not found - Train the model first")
                self.classification_model = None

            if os.path.exists(segmentation_model_path):
                self.segmentation_model = CustomModel(
                    model_path=segmentation_model_path,
                    classes=self.classes,
                    device=self.device,
                    model_type="segmentation"
                )
            else:
                st.warning("⚠ segmentation model not found - Train the model first")
                self.segmentation_model = None

        except Exception as e:
            st.error(f"Error loading models: {e}")
            self.classification_model = None
            self.segmentation_model = None

    def setup_audio(self):
        try:
            self.sound_manager = SoundManager()
        except Exception as e:
            st.error(f"Error during audio initialization: {e}")
            self.sound_manager = None

    def classify_image_with_audio(self, image):
     if self.classification_model is None:
        st.error("❌ Classification model not available")
        return None
     try:
        result = self.classification_model.predict(image)
        if result:
            class_name = result["class"]
            confidence = result["confidence"]
            
            description_text = f"The image contains a {class_name} with {int(confidence * 100)} percent confidence."
            
            from gtts import gTTS
            description_audio_buffer = BytesIO()
            tts = gTTS(text=description_text, lang='en')
            tts.write_to_fp(description_audio_buffer)
            description_audio_buffer.seek(0)
            
            if self.sound_manager:
                object_sound = self.sound_manager.get_sound_for_class(class_name)
            else:
                object_sound = None
            
            return {
                "class": class_name, 
                "confidence": confidence, 
                "text": description_text,
                "description_audio": description_audio_buffer,
                "object_sound": object_sound
            }
        return None
     except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

    def segment_image_with_enhanced_mask(self, image):
        if self.segmentation_model is None:
            st.error("❌ Segmentation model not available")
            return None
        try:
            result = self.segmentation_model.predict(image)
            if result:
                enhanced_mask = self.create_enhanced_colored_mask(
                    result["segmentation_map"], 
                    image.size
                )
                
                segmentation_audio = self.generate_segmentation_audio(result["detected_classes"])
                
                return {
                    "segmentation_map": result["segmentation_map"],
                    "enhanced_colored_mask": enhanced_mask,
                    "original_colored_mask": result["colored_mask"],
                    "detected_classes": result["detected_classes"],
                    "class_distribution": result["class_distribution"],
                    "segmentation_audio": segmentation_audio
                }
            return None
        except Exception as e:
            st.error(f"Erreur lors de la segmentation: {e}")
            return None

    def generate_segmentation_audio(self, detected_classes):
        try:
            if not detected_classes:
                text = "No objects detected in the segmentation."
            elif len(detected_classes) == 1:
                text = f"One object detected: {detected_classes[0]}."
            else:
                class_list = ", ".join(detected_classes[:-1]) + f" and {detected_classes[-1]}"
                text = f"{len(detected_classes)} objects detected: {class_list}."
            
            from gtts import gTTS
            audio_buffer = BytesIO()
            tts = gTTS(text=text, lang='en')
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception as e:
            st.error(f"Error generating segmentation audio: {e}")
            return None

    def create_enhanced_colored_mask(self, segmentation_map, original_size):
        class_colors = {
            0: [0, 0, 0],       
            1: [255, 50, 50],     
            2: [50, 255, 50],    
            3: [50, 50, 255],   
            4: [255, 255, 50], 
            5: [255, 50, 255], 
        }
        if segmentation_map.shape[:2] != (original_size[1], original_size[0]):
            segmentation_resized = cv2.resize(
                segmentation_map.astype(np.uint8), 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            segmentation_resized = segmentation_map

        height, width = segmentation_resized.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id in range(len(self.classes) + 1):  
            if class_id in class_colors:
                mask = segmentation_resized == class_id
                colored_mask[mask] = class_colors[class_id]
 
        colored_mask = cv2.GaussianBlur(colored_mask, (3, 3), 0) 
        non_black_mask = np.any(colored_mask != [0, 0, 0], axis=2)
        colored_mask[non_black_mask] = np.clip(
            colored_mask[non_black_mask] * 1.2, 0, 255
        ).astype(np.uint8)
        return Image.fromarray(colored_mask)

    def train_models(self, model_type): 
        try:
            if model_type == "classification":
                trainer = ModelTrainer(num_epochs=10, batch_size=16)
                trainer.train_classification_model() 
            elif model_type == "segmentation":
                trainer = ModelTrainer(num_epochs=10, batch_size=8)
                trainer.train_segmentation_model() 
            self.setup_models()
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement: {e}")

def play_two_audios_in_streamlit(description_audio, object_sound, labels=["Description Audio", "Object Sound"]):
    col1, col2 = st.columns(2)
    with col1:
        if description_audio:
            st.markdown(f"#### 🔊 {labels[0]}:")
            st.audio(description_audio, format='audio/mp3')
    with col2:
        if object_sound:
            st.markdown(f"#### 🔊 {labels[1]}:")
            st.audio(object_sound, format='audio/mp3')

def main(): 
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>  IA Vision & Audio</h1>
        <p style='font-size: 18px; color: #666;'>Classification • Segmentation avec Audio</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
 
    if "app" not in st.session_state:
        with st.spinner("  Initializing the application..."):
            st.session_state.app = ImageProcessingApp()
    app = st.session_state.app
 
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "  Home" 
    st.sidebar.markdown(" Menu Principal") 
    menu_options = [
        ("  Home", "home"),
        ("  Classification", "classification"), 
        ("  Segmentation", "segmentation") 
    ]
 
    for option_text, option_key in menu_options: 
        if st.session_state.selected_mode == option_text: 
            st.sidebar.button(
                option_text, 
                key=f"menu_{option_key}",
                use_container_width=True,
                disabled=True,
                help="Mode actuellement sélectionné"
            )
        else: 
            if st.sidebar.button(
                option_text, 
                key=f"menu_{option_key}",
                use_container_width=True
            ):
                st.session_state.selected_mode = option_text
                st.rerun()

    st.sidebar.markdown("---") 
   
    mode = st.session_state.selected_mode
 
    if mode == "  Home":
        st.markdown("##  Welcome to the AI ​​Vision & Audio app")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ###   Classification
            -Identifies the main object in the image
            - Generates an audio description
            - Provides a confidence score
            """)
        with col2:
            st.markdown("""
            ###  Segmentation
            - Precisely segments objects
            - Displays a vivid color mask
            - Provides detailed statistics
            """)
        st.markdown("---") 

    else: 
        st.markdown(" Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file) 
            st.markdown(" Original Image")
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(image, caption="Uploaded image", use_container_width=True)
            st.markdown("---")

            if mode == "  Classification": 
             if st.button(" Classify the Image", key="classify_main", use_container_width=True):
              with st.spinner("  Classification in progress..."):
               result = app.classify_image_with_audio(image)
               if result: 
                    play_two_audios_in_streamlit(
                        result['description_audio'],
                        result['object_sound']
                    )
               else:
                 st.error("❌Error during classification")

            elif mode == "  Segmentation":  
                if st.button(" Segment the Image", key="segment_main", use_container_width=True):
                    with st.spinner("  Segmentation in progress..."):
                        result = app.segment_image_with_enhanced_mask(image)
                        if result: 
                            st.markdown("  Colored Segmentation Mask") 
                            st.image(result["enhanced_colored_mask"],  
                                    use_container_width=True) 
                            st.markdown("---")
                            
                            if result["segmentation_audio"]:
                                st.markdown("#### 🔊  ")
                                st.audio(result["segmentation_audio"], format='audio/mp3')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if result["detected_classes"]:
                                    st.markdown(" Detected Classes")
                                    for class_name in result["detected_classes"]: 
                                        color_emoji = {
                                            "cat": "🔴",
                                            "dog": "🟢", 
                                            "man": "🔵",
                                            "woman": "🟡",
                                            "rat": "🟣"
                                        }
                                        emoji = color_emoji.get(class_name, "⚪")
                                        st.write(f"{emoji} **{class_name}**")
                                else:
                                    st.warning("Aucune classe détectée")
                            with col2:
                                if result["class_distribution"]:
                                    st.markdown(" Class Distribution")
                                    for class_name, stats in result["class_distribution"].items(): 
                                        color_emoji = {
                                            "cat": "🔴",
                                            "dog": "🟢", 
                                            "man": "🔵",
                                            "woman": "🟡",
                                            "rat": "🟣"
                                        }
                                        emoji = color_emoji.get(class_name, "⚪")
                                        st.write(f"{emoji} **{class_name}**: {stats['percentage']:.1f}%")
                                else:
                                    st.warning("No distribution available")
                        else:
                            st.error("❌ Error during segmentation")
        else:
            st.info("Please upload an image to use this processing mode.")

if __name__ == "__main__":
    main()