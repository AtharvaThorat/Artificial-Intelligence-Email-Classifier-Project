
# 📬 Artificial Intelligence Email Classifier Project

**A machine learning system for classifying emails into Spam vs Ham using text processing and multiple classification models.**

This project demonstrates a complete NLP classification pipeline — from raw text preprocessing to model training, evaluation, and performance comparison.

---

## 🔍 Project Overview

Email spam filtering and classification is a core problem in natural language processing and cybersecurity. This project builds a machine learning-based classifier that automatically distinguishes **spam** emails from genuine (ham) emails using statistical and neural models.

The solution includes:

* Text preprocessing and feature extraction
* Multiple model implementations and comparison
* Metrics-based evaluation of classification performance

**Primary models used:**
✔ Multinomial Naive Bayes
✔ Bernoulli Naive Bayes
✔ Multi-Layer Perceptron (MLP) Classifier
(All standard supervised ML algorithms suitable for text classification) ([GitHub][1])

---

## 🧠 Problem Statement

Email communication is ubiquitous, yet filtering unwanted spam messages remains a persistent challenge. Efficient spam classifiers support:

✔ Email security
✔ User experience
✔ Automated filtering
✔ Reducing phishing and malicious messages

This project demonstrates an end-to-end machine learning pipeline to solve the binary classification task:

> *“Given the text of an email, classify it as SPAM or HAM (not spam).”* ([GitHub][1])

---

## 🚀 Key Features & Technical Scope

### ✅ Data Processing

* Removes irrelevant columns
* Encodes labels (`spam` → 1, `ham` → 0)
* Generates additional analytical features (e.g., message length)
* Converts text to numerical vectors using **CountVectorizer (Bag of Words)** ([GitHub][1])

---

### ✅ Machine Learning Models

| Model                       | Type                       | Strength                                     |               |
| --------------------------- | -------------------------- | -------------------------------------------- | ------------- |
| **Bernoulli Naive Bayes**   | Probabilistic Classifier   | Effective for binary/boolean text features   |               |
| **Multinomial Naive Bayes** | Probabilistic Classifier   | Well-suited to frequency-based text features |               |
| **MLP Classifier**          | Feedforward Neural Network | Captures non-linear patterns                 | ([GitHub][1]) |

Each model is trained and evaluated to compare performance using accuracy, precision, recall, and F1-score. ([GitHub][1])

---

## 🛠 Machine Learning Workflow

The end-to-end process implemented in the notebook includes:

1. **Loading & Cleaning Data**
   Remove noise and irrelevant metadata.

2. **Label Encoding**
   Convert textual categories to numeric classes.

3. **Feature Extraction – Bag of Words**
   Represent text as numeric vectors.

4. **Train/Test Split**
   Standard split for model evaluation.

5. **Model Training & Evaluation**
   Train classifiers and evaluate on test data with metrics.

6. **Performance Comparison**
   Compare models via accuracy and confusion matrices. ([GitHub][1])

---

## 📈 Results Summary

* All models show **high classification performance** on the dataset.
* Naive Bayes classifiers perform extremely well given the textual feature representation.
* Neural network (MLP) provides competitive results, validating non-linear learning capability. ([GitHub][1])

> The use of strong statistical baselines like Naive Bayes is appropriate given the problem’s text distribution and class imbalance — a common technique in NLP classification tasks. ([Wikipedia][2])

---

## 🛠 Tech Stack & Tools

| Category      | Tools                          |               |
| ------------- | ------------------------------ | ------------- |
| Language      | Python                         |               |
| Notebook      | Jupyter                        |               |
| NLP           | Scikit-learn (CountVectorizer) |               |
| ML Models     | Naive Bayes, MLPClassifier     |               |
| Data Handling | pandas, NumPy                  |               |
| Visualization | matplotlib, seaborn            | ([GitHub][1]) |

---

## 🧪 Evaluation Metrics

Typically, classification performance is measured using:

| Metric        | Purpose                       |               |
| ------------- | ----------------------------- | ------------- |
| **Accuracy**  | Overall correctness           |               |
| **Precision** | Spam detection exactness      |               |
| **Recall**    | Success at identifying spam   |               |
| **F1 Score**  | Balance of precision & recall | ([GitHub][1]) |

Reporting these metrics provides **interpretability and reliability**, both critical in ML model assessment.

---

## 🧩 Skills Demonstrated

This project highlights:

🎯 Natural Language Processing (NLP) fundamentals
🎯 Feature Engineering for text data
🎯 Supervised machine learning model training
🎯 Model evaluation and comparison
🎯 Practical dataset handling and experimentation
🎯 Quick prototyping in Jupyter Notebook

These are directly relevant to roles in:

* **Machine Learning Engineering**
* **AI Engineering**
* **Data Science**
* **Software Engineering (with ML focus)**
* **Natural Language Processing roles**

---

## 🧠 Why Recruiters Should Care

This is not just a classification script — it shows:

✔ End-to-end ML pipeline thinking
✔ Data preprocessing best practices
✔ Comparative evaluation of models
✔ Applied NLP techniques
✔ Clear result interpretation

These are real skills companies look for in **ML, AI, and NLP roles.**

---

## 🧾 Repository Structure

```
Artificial-Intelligence-Email-Classifier-Project/
│
├── (23,24,27)_AIES_MINI_PROJECT_EMAIL_CLASSIFIER.ipynb
├── README.md
└── (Dataset used inside notebook)
```

---

## ▶ How to Run

1. Clone the repo:

```
git clone https://github.com/AtharvaThorat/Artificial-Intelligence-Email-Classifier-Project
```

2. Open the Notebook:

```
jupyter notebook (23,24,27)_AIES_MINI_PROJECT_EMAIL_CLASSIFIER.ipynb
```

3. Install required packages:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

4. Run all cells to reproduce the model training and evaluation.

---

## 📬 About the Author

**Atharva Thorat**
Passionate about building applied AI and machine learning solutions.
Focus areas include:

* Machine Learning Models
* Natural Language Processing
* Data-Driven Decision Systems
* Scalable ML pipelines

