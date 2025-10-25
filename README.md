
---

# 📧 Email Spam Detection Using Machine Learning

### 🚀 Overview

This project aims to **classify emails as spam or ham (non-spam)** using machine learning models. With the increasing volume of digital communication, spam filtering has become crucial for both personal and professional safety. This system leverages **text preprocessing**, **Bag of Words representation**, and multiple classification algorithms to build an accurate and efficient spam detection model.

---

## 📊 Project Highlights

* **Dataset Used:** `spam.csv` (contains 5,572 labeled messages)
* **Primary Objective:** Automatically detect and classify emails as spam or ham.
* **Techniques Applied:**

  * Text preprocessing with `CountVectorizer` (Bag of Words model)
  * Model training using **Multinomial Naive Bayes**, **Bernoulli Naive Bayes**, and **Neural Network (MLPClassifier)**
  * Comparative performance evaluation with **accuracy**, **precision**, **recall**, and **F1-score** metrics

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Preprocessing

* Removed unnecessary columns and renamed relevant ones.
* Encoded categories:

  * `ham → 0`
  * `spam → 1`
* Created a `Length` column to analyze the distribution of message lengths.
* Visualized data using **Seaborn** and **Plotly** histograms to understand category distribution and message lengths.

### 2️⃣ Feature Extraction

* Employed **CountVectorizer** to convert text data into numerical feature vectors.
* Resulting feature matrix: `(5572 x 8672)` representing 8,672 unique tokens.

### 3️⃣ Train-Test Split

* Dataset divided into:

  * **Training Set:** 70% (3,900 samples)
  * **Testing Set:** 30% (1,672 samples)

---

## ⚙️ Models Implemented

| Model                               | Description                          | Accuracy   | Precision | Recall    | F1 Score  |
| ----------------------------------- | ------------------------------------ | ---------- | --------- | --------- | --------- |
| **Bernoulli Naive Bayes**           | Suitable for binary/boolean features | **0.9838** | **1.00**  | **0.873** | **0.932** |
| **Multinomial Naive Bayes**         | Best for text frequency features     | **0.9814** | **0.917** | **0.939** | **0.928** |
| **MLP Classifier (Neural Network)** | Multi-layer perceptron model         | **0.9800** | **0.970** | **0.900** | **0.930** |

All models demonstrated high accuracy and robustness in detecting spam, with Naive Bayes methods performing exceptionally well for textual data.

---

## 📈 Visual Analysis

### 🔹 Confusion Matrix Comparison

Displayed confusion matrices for all three models side-by-side using **Seaborn heatmaps**, showing strong true positive and true negative detection performance.

### 🔹 Metric Comparison Heatmap

A consolidated heatmap visualizing **Accuracy, Precision, Recall, and F1-Score** across all models for better interpretability.

---

## 🧪 Results and Insights

* The dataset contained **86.6% ham** and **13.4% spam** messages.
* **MultinomialNB** performed best overall, balancing precision and recall.
* The **BernoulliNB** achieved perfect precision, ensuring minimal false positives.
* **MLP Classifier** performed competitively but required more computation time.
* The analysis confirms that **CountVectorizer + MultinomialNB** provides a lightweight yet highly effective spam detection pipeline.

---

## 🧩 Tech Stack

| Category          | Tools & Libraries               |
| ----------------- | ------------------------------- |
| **Language**      | Python                          |
| **Data Handling** | Pandas, NumPy                   |
| **Visualization** | Matplotlib, Seaborn, Plotly     |
| **ML Algorithms** | Scikit-learn                    |
| **Environment**   | Jupyter Notebook / Google Colab |

---

## 🧰 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/email-spam-detection.git
cd email-spam-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook or Script

Open Jupyter Notebook or Google Colab and execute:

```python
python spam_detection.py
```

or

```bash
jupyter notebook Spam_Detection.ipynb
```

---

## 🔍 Future Enhancements

* Implement **TF-IDF Vectorizer** to improve feature representation.
* Integrate **deep learning models (LSTM / BERT)** for context-aware classification.
* Deploy the model as a **web or API service** for real-time email filtering.
* Expand dataset with multilingual and varied email sources for better generalization.

---

## 🏁 Conclusion

This project demonstrates how a well-structured text classification pipeline can effectively detect spam emails. By combining **CountVectorizer** with **Naive Bayes models**, the system achieves over **98% accuracy**, offering a reliable and scalable solution to spam filtering problems.

---

## 👨‍💻 Author

**Atharva Thorat**
📧 [[atharvathorat03@gmail.com](mailto:atharvathorat03@gmail.com)]
🔗 [GitHub Profile](https://github.com/AtharvaThorat)

---

