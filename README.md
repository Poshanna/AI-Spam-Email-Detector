# 📧 AI Spam Email Detector

An AI-powered web application that classifies SMS or email messages as **Spam** or **Not Spam (Ham)** using Machine Learning. The project uses **TF-IDF Vectorization** and a **Multinomial Naive Bayes** classifier, and provides predictions through an interactive **Streamlit** interface.

---

## 🚀 Features

* Detects whether a message is **Spam** or **Not Spam**
* Interactive web interface built with Streamlit
* Displays prediction confidence
* Uses TF-IDF for text feature extraction
* Machine Learning model trained using Multinomial Naive Bayes
* Clean and simple user interface
* Includes sample messages for testing

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas
* **Model Serialization:** Joblib
* **Web Framework:** Streamlit

---

## 📂 Project Structure

```text
AI-Spam-Email-Detector/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── spam.csv               # SMS Spam Collection dataset
├── model.pkl              # Trained ML model
├── vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Poshanna/AI-Spam-Email-Detector.git
```

### 2. Navigate to the project

```bash
cd AI-Spam-Email-Detector
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run:

```bash
python train_model.py
```

This will:

* Load and preprocess the dataset
* Train the spam detection model
* Save:

  * `model.pkl`
  * `vectorizer.pkl`

---

## 🌐 Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal (typically **http://localhost:8501**) in your browser.

---

## 📊 Machine Learning Workflow

1. Load SMS Spam Collection Dataset
2. Remove unnecessary columns
3. Rename dataset columns
4. Convert labels (`ham` → `0`, `spam` → `1`)
5. Split dataset into training and testing sets
6. Convert text into numerical features using TF-IDF
7. Train a Multinomial Naive Bayes classifier
8. Evaluate model performance
9. Save the trained model and vectorizer
10. Predict whether new messages are Spam or Not Spam

---

## 📈 Model Performance

**Algorithm:** Multinomial Naive Bayes

**Feature Extraction:** TF-IDF Vectorization

**Accuracy:** **96.68%**

### Classification Report

| Metric    |  Ham | Spam |
| --------- | ---: | ---: |
| Precision | 0.96 | 1.00 |
| Recall    | 1.00 | 0.75 |
| F1-Score  | 0.98 | 0.86 |

---

## 💡 Sample Messages

### 🚨 Spam

```text
Congratulations!

You have won ₹50,000.

Click here now to claim your prize.
```

### ✅ Not Spam

```text
Hi,

Can we meet after class today?

Thanks.
```

---

## 📸 Application Features

* Message input area
* Spam/Not Spam prediction
* Prediction confidence percentage
* Progress bar for confidence
* Example Spam and Ham messages
* Sidebar with project information

---

## 📚 Future Improvements

* Upload and classify `.txt` files
* Email attachment scanning
* Support for multiple languages
* Deep Learning (LSTM/BERT) implementation
* Email phishing detection
* Prediction history
* Cloud deployment
* Docker support

---

## 👨‍💻 Author

**Poshanna**

GitHub: https://github.com/Poshanna

---

## ⭐ Acknowledgements

* SMS Spam Collection Dataset
* Scikit-learn
* Streamlit
* Pandas
