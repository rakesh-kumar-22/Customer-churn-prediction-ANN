# Customer Churn Prediction using ANN 

This project is a **Customer Churn Prediction System** built using **Artificial Neural Networks (ANNs)**. The model predicts whether a customer is likely to churn based on various customer attributes. The app is deployed using **Streamlit** by a web application.

---

##  **Project Overview**
- Developed a **deep learning model (ANN)** using **TensorFlow/Keras**.
- Used **Scikit-learn** for **data preprocessing, encoding, and scaling**.
- **Streamlit-based Web App** for real-time predictions.
- The model was trained on the **Churn_Modelling.csv** dataset.

---

## 📂 **Project Structure**

📁 customer_churn_prediction/
│── 📜 app.py                # Streamlit app for prediction
│── 📜 churn_experiments.ipynb  # Jupyter notebook with data preprocessing & model training
│── 📜 model.h5              # Trained ANN model
│── 📜 scaler.pkl            # Scaler for feature normalization
│── 📜 label_encoder.pkl     # Label Encoder for Gender
│── 📜 onehotencoder.pkl     # OneHotEncoder for Geography
│── 📜 README.md             # Project documentation
│── 📜 requirements.txt      # Dependencies for running the project
│── 📊 Churn_Modelling.csv   # Dataset (not included, must be downloaded separately)

---

## 🛠 **Tech Stack**
- **Programming Language**: Python  
- **Machine Learning Libraries**: TensorFlow, Keras, Scikit-learn  
- **Web App Deployment**: Streamlit  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  

🧠 Model Details

Architecture: Fully Connected Artificial Neural Network (ANN)

Layers:

Input Layer: 12 Features

Hidden Layers: 64 neurons (ReLU) → 32 neurons (ReLU)

Output Layer: 1 neuron (Sigmoid)

Loss Function: binary_crossentropy

Optimizer: adam

Evaluation Metrics: accuracy

⭐ Like this project?

If you found this project helpful, please give it a star ⭐ on GitHub!
