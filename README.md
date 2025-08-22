# 💧DrinkSafeAI - AI powered water safety predictor

---

## 🔍 Project Goal
DrinkSafeAI is designed to predict the **potability of water** using **machine learning and deep learning models**. It provides a user-friendly interface to input water quality parameters and instantly predicts whether the water is safe to drink. The goal of this project is to combine **data-driven insights** with a simple and interactive deployment platform.

---

## 📖 Overview  
The project leverages **real-world water quality datasets** to train and evaluate AI models. It uses a Streamlit-based web app for deployment, enabling real-time predictions. By integrating preprocessing, model training, and a deployed interface, DrinkSafeAI demonstrates how AI can assist in solving environmental and health-related problems.

---

## 🔄 **Project Workflow**

### **1️⃣ Data Preprocessing & EDA**
- **Data Inspection:** Loaded the dataset with Pandas for an initial inspection of its structure and types.  
- **EDA & Visualization:** Visualized feature distributions and correlations using Seaborn and Matplotlib.  
- **Missing Value Imputation:** Filled null values for `ph`, `Sulfate`, and `Trihalomethanes` using the mean.  
- **Feature Scaling:** Normalized the feature set using `StandardScaler` to standardize the data range.  
- **Train-Test Split:** Divided the processed data into training and testing sets for model validation.  

---

### **2️⃣ Model Building**
Extensively tested and compared a wide range of models to find the best performer:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machines (SVM)  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  
- A **Convolutional Neural Network (CNN)** using TensorFlow/Keras was chosen as the final model for deployment.  

---

### **3️⃣ Evaluation Metrics**
- **Accuracy** was used as the key metric to evaluate and compare the performance of each model.  

---

### **4️⃣ Deployment**
- Built a **user-friendly web application** using Streamlit.  
- Integrated the **best-performing CNN model** (`model.keras`) and the corresponding **scaler** (`scaler.pkl`) for live predictions.  

---

## 🛠 **Tech Stack**

- Pandas / NumPy – For data manipulation and numerical operations.  
- Scikit-learn – For preprocessing and implementing baseline and ensemble models.  
- TensorFlow / Keras – For building the final CNN model.  
- Streamlit – To build and deploy the interactive web application.  
- Seaborn / Matplotlib – For data visualization.  
- Joblib – For saving and loading the scaler model.
  
---

## 📂 Project Structure  
```
DrinkSafeAI/
├── dataset/
│   └── water_condition.csv          # Raw dataset used for training
├── .gitignore                       # Files/directories to exclude from Git tracking
├── LICENSE                          # Allows reuse, with attribution, no warranty
├── README.md                        # Project documentation
├── app.py                           # Streamlit app script
drinking_water.ipynb             # Jupyter notebook for data processing and model training

├── model.keras                      # Trained deep learning model
├── scaler.pkl                       # Pre-fitted StandardScaler object for input normalization
├── requirements.txt                 # Project dependencies
└── 
```
---

## ✨ **Features**  

- Easy-to-use interactive interface  
- Real-time water potability prediction  
- Supports multiple water quality parameters  
- Displays probability score for better insights  

---

## 🚀 **Future Enhancements**  

- Cloud deployment with continuous updates  
- Addition of more advanced models like XGBoost and CatBoost  
- Integration with APIs for real-time water quality data  
- Mobile-friendly interface  

---
## 🧪 **How to Run Locally**

```
# Clone the repository
git clone https://github.com/yourusername/DrinkSafeAI.git

# Navigate to the project directory
cd DrinkSafeAI

# Install the dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

<p>
  <a href="mailto:aradhyaray99@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
  <a href="www.linkedin.com/in/rayaradhya"><img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
  <a href="https://github.com/AradhyaRay05"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

---

Thanks for visiting ! Feel free to explore my other repositories and connect with me. 🚀
