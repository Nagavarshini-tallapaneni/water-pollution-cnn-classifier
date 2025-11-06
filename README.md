# 🌊 Water Pollution Detection using CNN Model


## 📘 Project Overview

This project focuses on classifying **water body images** as **clean** or **polluted** using **Convolutional Neural Networks (CNNs)** to support sustainable water management and environmental monitoring.

* **Week 1**: Project setup, data preparation, and CNN base model using **MobileNetV2** (Transfer Learning).
* **Week 2**: Model fine-tuning, evaluation improvement, and training optimization for better results.



## Improvisations

* **Fine-Tuned MobileNetV2 Layers:** Unfroze top layers for improved feature learning specific to water images.
* **Hyperparameter Optimization:** Adjusted learning rate, batch size, and optimizer for better accuracy.
* **Evaluation Metrics Added:** Implemented precision, recall, F1-score, and confusion matrix visualization.
* **Callbacks Integration:** Added EarlyStopping and ReduceLROnPlateau to prevent overfitting and optimize training.
* **Performance Visualization:** Generated training and validation accuracy/loss graphs using Matplotlib.
* **Data Expansion:** Added more images and performed real-time augmentation for improved generalization.
* **Model Export:** Saved the fine-tuned model for Flask web integration in Week 3.



## 🧠 Model Architecture


MobileNetV2 (pretrained on ImageNet)
   ↓
GlobalAveragePooling2D
   ↓
Dense(128, activation='relu')
   ↓
Dropout(0.3)
   ↓
Dense(1, activation='sigmoid')


* **Base Model:** MobileNetV2
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam (lr = 1e-4)
* **Metrics:** Accuracy, Precision, Recall, F1-score



## 📊 Dataset Structure


data/
├── train/
│   ├── clean/
│   └── polluted/
└── val/
    ├── clean/
    └── polluted/


* Images resized to **224×224 px**
* Data normalized (`rescale=1./255`)
* Augmented using rotation, zoom, and horizontal flip



## ⚙️ How to Run

```bash
# Step 1: Clone this repository
git clone https://github.com/your-repo.git
cd your-repo

# Step 2: Install dependencies
pip install tensorflow matplotlib scikit-learn

# Step 3: Train and evaluate model
python train_finetune_model.py


## 📈 Results and Evaluation

After fine-tuning, validation accuracy improved notably compared to Week 1.

| Metric    | Week 1 |  Week 2  |
| :-------- | :----: | :------: |
| Accuracy  |  ~88%  |  **94%** |
| Precision |  0.87  | **0.93** |
| Recall    |  0.86  | **0.92** |

**Visualizations:**

* Accuracy vs. Validation Accuracy
* Loss vs. Validation Loss
* Confusion Matrix for prediction results


## 🌱 Future Scope 

* Develop a **Flask web app** for real-time image classification.
* Allow users to upload images and get predictions instantly.
* Deploy on a cloud service (Heroku/Render).




**Internship:** Edunet Foundation – AI for Sustainability
**Week 2 Milestone Submission ✅**
