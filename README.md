# heart_disease-risk-predictor

# HeartGuard AI - Cardiovascular Disease Risk Predictor 🫀
[Streamlit](https://heartdisease-risk-predictor.streamlit.app/)

This is a machine learning mini-project designed to predict the likelihood of heart disease in patients based on various clinical parameters. The project features a trained Random Forest model and an interactive web application built with Streamlit.

## 🌟 Features
* **Machine Learning Pipeline:** Data exploration, preprocessing, scaling, and model training using `scikit-learn`.
* **High Accuracy:** Utilizes a Random Forest Classifier achieving approximately 84% accuracy on test data.
* **Interactive Web Interface:** A user-friendly Streamlit dashboard that allows users to adjust patient parameters using sliders and dropdowns to see real-time risk predictions.
* **Medical Glossary:** Includes a built-in guide explaining medical features like `oldpeak`, `thalach`, and `trestbps`.

## 🛠️ Technologies Used
* **Programming Language:** Python 3
* **Data Processing & Analysis:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Web Deployment:** `streamlit`
* **Model Serialization:** `joblib`

## 📊 Dataset
The model is trained on the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). It uses 13 clinical features to predict the presence (1) or absence (0) of heart disease:
1. `age` - Age in years
2. `sex` - (1 = male; 0 = female)
3. `cp` - Chest pain type (0-3)
4. `trestbps` - Resting blood pressure (mm Hg)
5. `chol` - Serum cholesterol (mg/dl)
6. `fbs` - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. `restecg` - Resting electrocardiographic results (0-2)
8. `thalach` - Maximum heart rate achieved
9. `exang` - Exercise-induced angina (1 = yes; 0 = no)
10. `oldpeak` - ST depression induced by exercise relative to rest
11. `slope` - The slope of the peak exercise ST segment (0-2)
12. `ca` - Number of major vessels (0-4) colored by fluoroscopy
13. `thal` - Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)

## 🚀 Installation and Setup

### 1. Clone the repository
```bash
git clone <your-github-repo-link>
cd <repository-folder>
