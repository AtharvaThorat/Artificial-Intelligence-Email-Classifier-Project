

# 📬 Artificial Intelligence Email Classifier

A machine learning-based NLP system that classifies emails as **Spam** or **Ham** using statistical and neural classification models.

This project implements an end-to-end text classification pipeline — from preprocessing raw email text to feature engineering, model training, and performance evaluation.

---

## 🔍 Project Overview

Spam detection is a foundational problem in natural language processing and cybersecurity. This project builds and compares multiple supervised learning models to automatically classify email messages based on textual content.

The system includes:

* Text preprocessing and cleaning
* Feature extraction using Bag-of-Words
* Supervised classification models
* Quantitative performance evaluation

---

## 🧠 Problem Definition

Given the raw text of an email:

> Classify it as either **SPAM** or **HAM (not spam)**.

The challenge lies in transforming unstructured text into meaningful numerical representations that machine learning algorithms can learn from.

---

## 🛠 Technical Implementation

### 1️⃣ Data Preprocessing

* Removal of irrelevant columns
* Label encoding (`spam → 1`, `ham → 0`)
* Text cleaning
* Feature augmentation (e.g., message length analysis)

---

### 2️⃣ Feature Engineering

Text is converted into numerical vectors using:

**CountVectorizer (Bag of Words model)**

This approach:

* Represents word frequency
* Creates a sparse feature matrix
* Enables probabilistic and neural models to operate on text

---

### 3️⃣ Models Implemented

| Model                       | Type           | Why Used                                          |
| --------------------------- | -------------- | ------------------------------------------------- |
| **Multinomial Naive Bayes** | Probabilistic  | Strong baseline for frequency-based text features |
| **Bernoulli Naive Bayes**   | Probabilistic  | Effective for binary word presence features       |
| **MLP Classifier**          | Neural Network | Captures non-linear relationships in text         |

Each model is trained and evaluated on a train/test split for fair comparison.

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

These metrics provide insight into:

* False positive rates (spam misclassification)
* False negative rates (missed spam)
* Overall classification reliability

---

## 🏗 ML Workflow

```
Raw Email Dataset
        │
        ▼
Text Cleaning & Label Encoding
        │
        ▼
Feature Extraction (Bag of Words)
        │
        ▼
Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Performance Evaluation
```

This reflects a standard, production-style supervised ML pipeline.

---

## 🛠 Tech Stack

* **Python**
* **Jupyter Notebook**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib / Seaborn**

---

## 🧩 Engineering Strengths Demonstrated

### For Machine Learning Roles

* NLP preprocessing pipeline
* Feature vectorization techniques
* Baseline vs neural model comparison
* Proper model evaluation

### For AI Engineering Roles

* Structured transformation of unstructured data
* Applied text classification
* Understanding of probabilistic vs neural approaches

### For Software Engineering Roles

* Organized workflow
* Clear modular pipeline stages
* Reproducible experimentation
* Clean dataset handling

---

## 📈 Key Insights

* Naive Bayes models perform strongly in text classification tasks due to independence assumptions aligning well with word-frequency representations.
* Neural networks can capture more complex patterns but require careful evaluation to avoid overfitting.
* Feature engineering significantly impacts classification performance.

---

## 📂 Repository Structure

```
Artificial-Intelligence-Email-Classifier-Project/
│
├── (23,24,27)_AIES_MINI_PROJECT_EMAIL_CLASSIFIER.ipynb
├── README.md
└── Dataset (used within notebook)
```

---

## ▶ How to Run

```bash
git clone https://github.com/AtharvaThorat/Artificial-Intelligence-Email-Classifier-Project
cd Artificial-Intelligence-Email-Classifier-Project
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook (23,24,27)_AIES_MINI_PROJECT_EMAIL_CLASSIFIER.ipynb
```

Run all cells sequentially to reproduce preprocessing, training, and evaluation.

---

## 🎯 Why This Project Matters

This project demonstrates:

* End-to-end NLP pipeline design
* Applied machine learning in cybersecurity context
* Comparative model experimentation
* Practical understanding of evaluation metrics

These are directly relevant to:

* Machine Learning Engineer
* AI Engineer
* NLP Engineer
* Software Engineer (ML-focused roles)

---

## 👨‍💻 Author

**Atharva Thorat**
Master’s in Computer Science – University of Southern California
Focused on AI systems, machine learning pipelines, and applied NLP solutions.

