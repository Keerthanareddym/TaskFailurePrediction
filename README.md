# 🚀 Task Failure Prediction in Cloud Data Centres using Hybrid Learning Paradigms
This project is about “Task failure prediction in cloud data centers using Hybrid Learning Paradigms" where we can predict the failures of task/job with predictions using deep learning models.


## 📌 Project Overview

In large-scale cloud data centres, millions of tasks run every day, and task failures due to hardware issues, software bugs, or resource conflicts can cause serious service disruptions. This project leverages **Hybrid Deep Learning** (CNN + BiLSTM) models to **predict task failures before they occur**, improving reliability, efficiency, and fault tolerance in cloud environments.


## 🧠 Technologies Used

- Python 3.7
- TensorFlow & Keras
- Scikit-learn
- Flask (for Web Interface)
- SQLite (for database)
- Pandas, NumPy, Matplotlib, Seaborn

---
## 🏗️ Project Structure
Ah, I see the problem! That is a very common formatting issue. Thank you for sharing the screenshot.

The project structure looks like one long line because GitHub isn't rendering the line breaks correctly. To fix this, you need to tell Markdown to treat it as a pre-formatted code block.

Here is the corrected code. Notice the triple backticks (```) at the beginning and end.

Solution: Use a Code Block
Go back and edit your README.md file on GitHub.
Delete the entire "Project Structure" section that looks broken.
Copy and paste the code below to replace it.
✅ Copy this corrected block:



## 🏗️ Project Structure

```
├── model.py         # Handles data preprocessing, training, and evaluation
├── app.py           # Flask web app (signup, predict, SQLite integration)
├── dataset/         # Contains the dataset file(s) used for training
├── models/          # Stores the saved, trained models (CNN+BiLSTM, etc.)
├── templates/       # HTML templates (home, signup, signin, predict, result)
├── static/          # Static files (CSS, JS, images - if any)
└── README.md        # This documentation file

```
---
## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<Keerthanareddym>/<TaskFailurePrediction>.git
cd <TaskFailurePrediction>
```

### 2: (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
```
### 3. Install Required Packages

```bash
pip install -r requirements.txt
pip install flask scikit-learn pandas numpy tensorflow imbalanced-learn joblib
```
--- 

## 🛠️ How to Run the Project

### Step 1: Train the Model
```bash
python model.py
```
-Loads and cleans dataset
-Trains ML models (RandomForest, DecisionTree, VotingClassifier)
-Trains CNN+BiLSTM deep learning model
-Saves models to models/ folder

### Step 2: Launch Web Application
```bash
python app.py
```
-Open in browser: http://127.0.0.1:5000/
-Sign up / Log in
-Enter task parameters to get failure prediction result

---
## 🌐 Web Interface Features

--User Authentication (Sign up & Login)
--Input Features for Prediction:
    Time
    Instance Events Type
    Scheduling Class
    Priority
--AI-based task failure prediction
--Prediction result page with confidence output
--Navigation to performance graphs and model metrics

---

## 📊 Model Performance Comparison
| Model             | Accuracy | Precision | Recall | F1 Score |
| ----------------- | -------- | --------- | ------ | -------- |
| CNN + BiLSTM      | 0.97     | 0.92      | 0.98   | 0.95     |
| Random Forest     | 0.95     | 0.91      | 0.94   | 0.93     |
| Decision Tree     | 0.94     | 0.91      | 0.93   | 0.93     |
| Voting Classifier | 0.95     | 0.91      | 0.94   | 0.93     |

![Accuracy Graph](![accuracy_comparison](https://github.com/user-attachments/assets/e865e60e-d11d-48f4-921e-bfcd762544dc)
)


---
## 📈 Future Enhancements

Integrate Transformer-based models like BERT/GPT/ViT
Real-time streaming prediction (using Kafka, etc.)
Deploy to cloud using Docker, Kubernetes
Visual dashboards using SHAP/LIME for interpretability
Enhance scalability using distributed training

---
## 📎 References
Google Cluster Trace Dataset
IEEE Research Articles on CNN, BiLSTM, and Task Failure Prediction
