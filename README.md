

# 🔬 Refractive Index Prediction Using Machine Learning

This project predicts the **refractive index of materials based on their density** using multiple machine learning regression algorithms.
It was developed as part of an initiative under the **Photonics Society**, focusing on the correlation between material properties and optical behavior.

---

## 🚀 Project Overview

The project explores and compares different regression models to accurately estimate **refractive index ↔ density relationships**.
It also includes an **interactive prediction interface** and **visual performance analysis** through plots and evaluation metrics.

---

## 🧠 Objectives

* Predict refractive index from given density values.
* Reverse-predict density from known refractive index.
* Benchmark 20+ regression algorithms for performance.
* Visualize results using interactive and illustrative diagrams.

---

## ⚙️ Tech Stack

* **Languages:** Python
* **Libraries:**

  * `pandas`, `numpy`, `matplotlib` – Data handling and visualization
  * `scikit-learn` – ML models, metrics, and preprocessing
  * `xgboost`, `lightgbm`, `catboost` – Advanced regression models
  * `ipywidgets` – Interactive input/output interface

---

## 🧩 Machine Learning Models Used

* **Core Models:** SVR (Linear), Polynomial Regression, Extra Trees
* **Benchmark Models:**
  Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting,
  XGBoost, LightGBM, CatBoost, ANN, KNN, Decision Tree, Gaussian Process, etc.

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* **R² Score** – Model accuracy
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Square Error)**

All results are plotted as **Actual vs Predicted** graphs for visual comparison.

---

## 🖼️ Illustrative Visualizations

* Scatter plots of **Actual vs Predicted** values for each model
* Interactive prediction widgets using **ipywidgets** for real-time inference
* Benchmark results table comparing all models’ performance

---

## 💡 Key Features

* Interactive prediction of refractive index/density values
* Comparative benchmarking of 20+ ML models
* Visualized results with clean and informative plots
* Built for research under the **Photonics Society**

---

## 📁 Project Structure

```
📂 Refractive_Index_Prediction
│
├── germanatedata - Sheet1.csv     # Dataset
├── refractive_index_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 📈 Sample Output

* **R² Score**, **MAE**, and **RMSE** for each model printed in console
* Real-time predictions visible through Jupyter Notebook widgets
* Multiple regression models plotted for performance visualization

---

## 🧪 Future Enhancements

* Include deep learning models for better non-linear mapping
* Add web-based Streamlit interface for deployment
* Extend dataset with more optical material parameters

---

## 👨‍🔬 Author

**Karthik Kamath**
Member – *Photonics Society*
📍 Dayananda Sagar College of Engineering

---

Would you like me to make this **GitHub-ready** (formatted with emojis, markdown styling for code, and sections like “How to Run”)? It’ll look perfect on your repo.
