# 📊 Customer Churn Prediction

### 🚀 Project Overview
This project aims to predict **customer churn**—whether a customer will stop using a service—using machine learning models like **Logistic Regression** and **Random Forest**. The dataset contains customer information from a telecommunications company, including demographic details, contract information, payment methods, and other relevant features.

The goal is to build an accurate model that helps predict whether a customer will churn based on the available data.

### 📁 Dataset
The dataset used is the **Telco Customer Churn Dataset** and contains the following key features:

- **🆔 customerID**: Unique identifier for each customer (not used in prediction)
- **👫 gender**: Male or Female
- **👴 SeniorCitizen**: Indicates if the customer is a senior citizen (0 = No, 1 = Yes)
- **💍 Partner**: Whether the customer has a partner
- **👶 Dependents**: Whether the customer has dependents
- **📆 tenure**: Number of months the customer has been with the company
- **📞 PhoneService**: Whether the customer has phone service
- **📶 MultipleLines**: Whether the customer has multiple phone lines
- **🌐 InternetService**: Customer’s internet service provider (DSL, Fiber optic, or None)
- **🔐 OnlineSecurity, 💾 OnlineBackup, 🛡 DeviceProtection, 👨‍💻 TechSupport, 📺 StreamingTV, 🎬 StreamingMovies**: Whether the customer has these services
- **📄 Contract**: Customer's contract type (Month-to-month, One year, Two year)
- **📨 PaperlessBilling**: Whether the customer uses paperless billing
- **💳 PaymentMethod**: Payment method (Electronic check, Mailed check, etc.)
- **💲 MonthlyCharges**: The monthly amount charged to the customer
- **💵 TotalCharges**: Total amount charged to the customer
- **❌ Churn**: The target variable (Yes = customer churned, No = customer did not churn)

### 🔗 Dataset Source
The dataset for this project is available publicly on Kaggle. You can access it here:
[👉 Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

### ⚙️ Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install required dependencies**:
   Use the `requirements.txt` to install all necessary Python libraries.
   ```bash
   pip install -r requirements.txt
   ```

---

### 🛠 Usage

1. **Open the notebook**:
   Open the `churn_prediction.ipynb` file in **Google Colab** or your favorite Jupyter Notebook environment.

2. **Step-by-Step Process**:
   - Load and clean the dataset 📂
   - Preprocess the data (encoding categorical variables, handling missing values) 🧹
   - Train machine learning models (Logistic Regression, Random Forest) 🎯
   - Tune hyperparameters using **GridSearchCV** 🛠
   - Visualize feature importance 🔍

3. **Model Output**:
   After running the models, you'll get detailed results like:
   - **✅ Accuracy Score**
   - **🔄 Confusion Matrix**
   - **📊 Classification Report**
   - **🔥 Feature Importance Plot**

---

### 🔍 Models Used

- **⚙️ Logistic Regression**: A standard baseline model for binary classification.
- **🌲 Random Forest**: A powerful ensemble method providing higher accuracy.
- **🔧 GridSearchCV**: Used for hyperparameter tuning of the Random Forest model.

---

### 📊 Key Metrics

- **🎯 Accuracy**: The proportion of correctly classified instances.
- **🔄 Confusion Matrix**: Measures model performance in terms of **True Positives**, **False Positives**, **False Negatives**, and **True Negatives**.
- **📈 Precision, Recall, F1-Score**: Additional metrics focusing on the balance between precision and recall.

---

### 🔥 Sample Outputs

- **📊 Logistic Regression Accuracy**: ~80-82%
- **🌲 Random Forest Accuracy**: ~82-85%
- **🔄 Confusion Matrix**: Displays how the model performed on test data.
- **📉 Feature Importance**: Highlights the most impactful features like **MonthlyCharges**, **tenure**, and **Contract type**.
