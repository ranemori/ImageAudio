# AI Vision & Audio Processing System

A comprehensive Streamlit-based application for image classification and segmentation with automated audio generation capabilities. This system leverages custom deep learning models to identify and segment objects in images, subsequently generating descriptive audio content and corresponding sound effects.

## Overview

The AI Vision & Audio Processing System is designed for real-time image analysis with multi-modal output generation. The application combines computer vision techniques with audio synthesis to provide an accessible and interactive experience for image understanding tasks.

### Key Capabilities

- **Image Classification**: Object identification with confidence scoring
- **Semantic Segmentation**: Pixel-wise object detection and boundary delineation  
- **Audio Generation**: Automated text-to-speech synthesis and object-specific sound effects
- **Real-time Processing**: Streamlined inference pipeline for immediate results
- **Interactive Interface**: Web-based UI with intuitive navigation and visualization

## System Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Image Input   │ -> │  Model Pipeline  │ -> │  Audio Output   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Preprocessing  │    │  Classification  │    │      TTS        │
│   & Validation  │    │  Segmentation    │    │  Sound Effects  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Model Architecture

- **Classification Model**: MobileNetV2 with custom classification head
- **Segmentation Model**: DeepLabV3 with MobileNetV3 backbone
- **Audio Processing**: Google Text-to-Speech (gTTS) integration
- **Sound Management**: Custom audio file handling and playback system

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, recommended for optimal performance)
- Minimum 8GB RAM
- 2GB available disk space

### Environment Setup

```bash 
git clone https://github.com/your-organization/ai-vision-audio.git
cd ai-vision-audio
 
python -m venv venv
 
source venv/bin/activate
 
venv\Scripts\activate

 
pip install -r requirements.txt
```

### Dependencies
 
requirement.txt 

## Project Structure

```
Image-To-Audio/
├── app.py                      # Main Streamlit application
├── model.py                    # Custom model implementations
├── image_recognition.py        # Image processing utilities
├── sound_manager.py           # Audio management system
├── train_model.py             # Model training pipeline
├── segmentation_model.py   
├── mobilenet_finetuned.pth
├── segmentation_model.pth
├── sounds/               # Audio asset library
│   ├── cat_meow.mp3
│   ├── dog_bark.mp3
│   ├── rat_squeak.mp3
│   ├── man_hello.mp3
│   └── woman_hello.mp3
├── dataset/                # data asset library
│   ├── train/               
│   │   ├── annotations/
│   │   │   └──  train_annotations.json
│   │   ├── images/
│   │   │   ├── cat/ (images)
│   │   │   ├── dog/ (images)
│   │   │   ├── man/ (images)
│   │   │   ├── rat/ (images)
│   │   │   └── woman/ (images)
│   │   └──  masks/ (images)
│   ├──  val/
│   │   ├── annotations
│   │   │   └──  val_annotations.json
│   │   ├── images/
│   │   │   ├── cat/ (images)
│   │   │   ├── dog/ (images)
│   │   │   ├── man/ (images)
│   │   │   ├── rat/ (images)
│   │   │   └── woman/ (images)
│   │   └── masks/ (images)
│   ├── dataset_organizer.py 
│   └── mask_generator.py
├── segmentation_training_info.json
├── classification_training_info.json
├── icon.png
├── requirements.txt
├── config.yaml
└── README.md
```

## Usage

### Operational Modes

#### Classification Mode

1. **Image Upload**: Support for PNG, JPG, JPEG formats (max 10MB)
2. **Processing**: Automated preprocessing and model inference
3. **Results**: Class prediction with confidence score
4. **Audio Output**: Generated description and object-specific sounds

#### Segmentation Mode

1. **Image Input**: Same format support as classification
2. **Segmentation**: Pixel-wise object boundary detection
3. **Visualization**: Color-coded segmentation masks
4. **Analytics**: Detailed class distribution statistics
5. **Audio Feedback**: Comprehensive object detection summary

## Model Configuration

### Classification Model Specifications

- **Architecture**: MobileNetV2
- **Input Resolution**: 224×224 pixels
- **Output Classes**: 5 categories (cat, dog, man, woman, rat)
- **Inference Time**: ~0.3s (GPU) / ~2.0s (CPU)
- **Model Size**: ~14MB

### Segmentation Model Specifications

- **Architecture**: DeepLabV3 with MobileNetV3 backbone
- **Input Resolution**: 512×512 pixels  
- **Output Resolution**: Variable (maintains aspect ratio)
- **Classes**: Background + 5 object categories
- **Inference Time**: ~0.7s (GPU) / ~5.0s (CPU)
- **Model Size**: ~39MB

## Audio System

### Text-to-Speech Configuration

- **Engine**: Google Text-to-Speech (gTTS)
- **Languages**: Multi-language support
- **Voice Selection**: Gender-specific voice assignment
- **Output Format**: MP3, 22kHz sampling rate
 
### Supported Audio Formats

- Input: MP3, WAV, FLAC
- Output: MP3 (streaming compatible)
- Quality: 128kbps encoding

## Performance Benchmarks

### Memory Requirements

- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM
- **GPU Memory**: 4GB VRAM (recommended)

## Development

### Code Quality Standards

- **Style Guide**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Type Hints**: Full type annotation coverage
- **Testing**: Minimum 80% code coverage

### Contributing Guidelines

1. **Fork Repository**: Create personal fork of the project
2. **Feature Branch**: Create branch from `develop` 
3. **Implementation**: Follow coding standards and add tests
4. **Documentation**: Update relevant documentation
5. **Pull Request**: Submit PR with detailed description 

### Performance Optimization

- Enable mixed precision training: `--fp16`
- Use model compilation: `torch.compile(model)`
- Implement batch processing for multiple images
- Cache model outputs for repeated inference

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{ai_vision_audio_2025,
  title={AI Vision \& Audio Processing System},
  author={Rania},
  year={2025},
  url={https://github.com/ranemori/ImageAudio.git}
}
```
## Support
config.yaml

### Documentation
- **Model Training Guide**: [training.md](training.md)


### Professional Support
- **Email**: raniarina.y@gmail.com  

---

**Version**: 1.0.0  
**Release Date**: May 2025  
**Compatibility**: Python 3.8+ | PyTorch 2.0+ | CUDA 11.8+
