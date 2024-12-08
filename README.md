# **Glaucoma Detection Using Deep Learning**  

---

## **📌 Problem Statement**  

Glaucoma is a chronic, irreversible eye disease that damages the optic nerve, potentially causing blindness if left untreated. Early detection is crucial, but traditional diagnostic methods require expert evaluation, which is expensive and time-consuming. Given its global impact, an automated deep learning-based diagnostic system can provide scalable, cost-effective solutions.  

This project leverages deep learning for glaucoma detection using retinal fundus images. The objective is to build a model that accurately classifies images as **Glaucoma Positive** or **Glaucoma Negative**, supporting healthcare professionals in early diagnosis and treatment.  

---

### **🎯 Key Objectives:**  

- **High Accuracy:** Maximize overall prediction accuracy.  
- **Low False Positives:** Avoid misclassifying healthy eyes.  
- **Low False Negatives:** Minimize undetected glaucoma cases.  

The deep learning model is based on a Vision Transformer (ViT), a state-of-the-art neural network architecture. This solution can assist in developing automated glaucoma screening systems for hospitals, clinics, and research institutions.  

---

<br>
<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![React Router](https://img.shields.io/badge/React_Router-CA4245?style=for-the-badge&logo=react-router&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

</div>
<br>


---

## **📁 Project Structure**  

```
Glaucoma-Detection/  
├── dataset_preparation.py     # Preprocess dataset and organize images  
├── train_model.py              # Train ViT model on the dataset  
├── train_plot.py               # Plot training results  
├── test.py                     # Evaluate model performance  
├── assets/                     # Contains plots and result images  
├── requirements.txt            # Required libraries  
└── README.md                   # Project overview (this file)  
```

---

## **🔧 Installation and Setup**  

1. **Clone the Repository:**  
   ```bash
   !git clone https://github.com/AN-06/Glaucoma-Detection.git
   ```

2. **Navigate to the Project Directory:**  
   ```bash
   %cd Glaucoma-Detection
   ```

3. **Install Required Libraries:**  
   ```bash
   !pip install -r requirements.txt
   ```

4. **Run the Project:**  
   ```bash
   !python main.py [arguments]
   ```

**Supported Arguments:**  
- `train_model` - Train the model from scratch  
- `existing` - Retrain an existing model  
- `make_predictions` / `None` - Use a pre-trained model for inference  

---

## **📊 Dataset Structure**  

```
DATASET/  
├── train/  
│   ├── Glaucoma_Positive/  
│   └── Glaucoma_Negative/  
├── val/  
│   ├── Glaucoma_Positive/  
│   └── Glaucoma_Negative/  
└── test/  
    ├── Glaucoma_Positive/  
    └── Glaucoma_Negative/  
```

📌 **Dataset Download Link:** [Google Drive Dataset](https://drive.google.com/drive/folders/1M89d5jKBInbhvmEC95zn51zD6A25HKbF?usp=share_link)  

---

## **💻 Model Overview**  

### **Model Architecture:**  
- **Base Model:** Vision Transformer (ViT)  
- **Fully Connected Layers:**  
  - Activation: GELU  
  - Final Layer: Softmax for binary classification  

### **Training and Evaluation Metrics:**  
- **Accuracy**  
- **F1-Score**  
- **Confusion Matrix**  

---

## **📈 Results**  

### **🔍 Accuracy Plot:**  

<div align="center">  

![Accuracy Plot](https://github.com/AN-06/Glaucoma-Detection/blob/main/assets/Unknown-13.png)  

</div>  

---

### **📷 Image Processing Overview:**  

<div align="center">  

![Image Processing](https://github.com/AN-06/Glaucoma-Detection/blob/main/assets/IMAGE%20PROCESSING.png)  

</div>  

---

## **📦 Dependencies**  

### **Core Libraries:**  
- `numpy` - Numerical operations  
- `pandas` - Data analysis and manipulation  
- `scikit-learn` - Evaluation metrics  

### **Deep Learning Libraries:**  
- `tensorflow` - Deep learning framework  
- `keras` - Model building interface  

### **Image Processing Libraries:**  
- `opencv-python` - Advanced image processing (optional)  
- `Pillow` - Basic image processing  

### **Visualization Libraries:**  
- `matplotlib` - Data and result visualization  
- `seaborn` - Enhanced plotting  

---

## **🚀 How to Train the Model**  

You can install the modules individually with pip install, or add them to a requirements.txt file:

- `pandas`
- `scikit-learn`
- `tensorflow`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `numpy`

  * Start by opening Google Colab.

* Ensure that you have access to a TPU by selecting `Runtime` > `Change runtime type` > `Hardware accelerator` > `TPU`.

* Clone the repository using the following command:

    ```bash
    !git clone https://github.com/AN-06/GlaucomaDetection.git
    ```

* Change into the cloned directory:

    ```bash
    %cd GlaucomaDetection
    ```

* Install the required modules by running:

    ```bash
    !pip install -r requirements.txt
    ```

* To train your model or make predictions, run the following command:

    ```bash
    !python main.py [arguments]
    ```

    **Arguments:**
    - `train_model` - Specify this if you want to train the model before inference.
    - `existing` - Use this after the `train_model` argument if you want to retrain an existing model.
    - `make_predictions` / `None` - This loads the existing model for inference.


---

## **📑 Additional Notes:**  

- Ensure that the dataset follows the correct folder structure before training.  
- Use a **TPU** or **GPU** for faster training and better performance.  
- Experiment with different preprocessing techniques if necessary.  

---






