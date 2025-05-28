# Project Setup Instructions

## Environment Setup
```bash 
pip install -r requirements.txt
cd (project directory)
```

## Adding Images or Classes
### Create annotations
```bash
python ./dataset/dataset_organizer.py
```
### Create masks
```bash
python ./dataset/mask_generator.py
```

## Train Models
### Classification 
```bash
python train_model.py --model classification --epochs 10 --batch_size 16 --learning_rate 0.001
```
### Segmentation
```bash
python train_model.py --model segmentation --epochs 10 --batch_size 16 --learning_rate 0.001
```
### Both
```bash
python train_model.py --model both --epochs 10 --batch_size 16 --learning_rate 0.001
```
## Launch Application
```bash
python -m streamlit run app.py
```
