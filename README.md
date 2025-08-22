# DrinkSafeAI - AI powered water safety predictor

---

## 🔍 Project Goal
DrinkSafeAI is designed to predict the **potability of water** using **machine learning and deep learning models**. It provides a user-friendly interface to input water quality parameters and instantly predicts whether the water is safe to drink. The goal of this project is to combine **data-driven insights** with a simple and interactive deployment platform.

---

## 📖 Overview  
The project leverages **real-world water quality datasets** to train and evaluate AI models. It uses a Streamlit-based web app for deployment, enabling real-time predictions. By integrating preprocessing, model training, and a deployed interface, DrinkSafeAI demonstrates how AI can assist in solving environmental and health-related problems.

---

🔄 **Project Workflow**

### **1️⃣ Data Preprocessing & EDA**
- **Data Inspection:** Loaded the dataset with Pandas for an initial inspection of its structure and types.  
- **EDA & Visualization:** Visualized feature distributions and correlations using Seaborn and Matplotlib.  
- **Missing Value Imputation:** Filled null values for `ph`, `Sulfate`, and `Trihalomethanes` using the mean.  
- **Feature Scaling:** Normalized the feature set using `StandardScaler` to standardize the data range.  
- **Train-Test Split:** Divided the processed data into training and testing sets for model validation.  

---

### **2️⃣ Model Building**
Extensively tested and compared a wide range of models to find the best performer:

**Baseline Models:**
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machines (SVM)  
- Naive Bayes  
- Decision Tree  

**Ensemble & Boosting Models:**
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

**Deep Learning:**
- A **Convolutional Neural Network (CNN)** using TensorFlow/Keras was chosen as the final model for deployment.  

---

### **3️⃣ Evaluation Metrics**
- **Accuracy** was used as the key metric to evaluate and compare the performance of each model.  

---

### **4️⃣ Deployment**
- Built a **user-friendly web application** using Streamlit.  
- Integrated the **best-performing CNN model** (`model.keras`) and the corresponding **scaler** (`scaler.pkl`) for live predictions.  

---

🛠 **Tech Stack**

**Programming Language:**  
- Python – Core programming language.  

**Data Science & ML:**  
- Pandas / NumPy – For data manipulation and numerical operations.  
- Scikit-learn – For preprocessing and implementing baseline and ensemble models.  
- TensorFlow / Keras – For building the final CNN model.  
- XGBoost – High-performance gradient boosting library.  
- LightGBM – Fast, distributed, high-performance gradient boosting framework.  
- CatBoost – Gradient boosting on decision trees with categorical features support.  

**Deployment & Visualization:**  
- Streamlit – To build and deploy the interactive web application.  
- Seaborn / Matplotlib – For data visualization.  
- Joblib – For saving and loading the scaler model.  



---

## 📬 Contact

<p>
  <a href="mailto:aradhyaray99@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
  <a href="www.linkedin.com/in/rayaradhya"><img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
  <a href="https://github.com/AradhyaRay05"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

---

Thanks for visiting ! Feel free to explore my other repositories and connect with me. 🚀
