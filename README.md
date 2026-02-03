# Fake News Detection Project

This project aims to develop a machine-learning model capable of identifying and classifying news articles as Fake or Not Fake. The spread of fake news can have serious adverse effects on society, culture, and public opinion.
This project involves building and training multiple machine learning models on a diverse dataset of news articles and comparing their performance to identify the most accurate classifier.

The following techniques were used to evaluate the model performance:

Logistic Regression

Decision Tree Classifier

Gradient Boosting Classifier

Linear Support Vector Machine (Linear SVM) ✅ (Improved Accuracy)


## Project Overview

Fake news has become a major challenge in the digital age, where information spreads rapidly through social media and online platforms. Manual verification of news is inefficient at scale.
This project leverages Natural Language Processing (NLP) and machine learning algorithms to automatically determine the authenticity of news articles and help combat misinformation.

## Dataset

We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
- True: Genuine news articles
- False: Fake or fabricated news articles

# MLOps Extension with MLflow

To extend the project beyond traditional model training and evaluation, MLflow was integrated into the workflow as part of the MLOps lifecycle. This enhancement enables experiment tracking, model versioning, and controlled model promotion.

# Model Improvement using Linear SVM

After training and evaluating all models, Linear SVM demonstrated better accuracy and F1-score compared to other classifiers.
This improvement is due to Linear SVM’s effectiveness in handling high-dimensional and sparse text data, especially when combined with TF-IDF features.

As a result, Linear SVM was selected as the final model for fake news classification.

## Dependencies

Before running the code, make sure you have the following libraries and packages installed:

- Python 3
- Scikit-learn
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Regular Expression

You can install these dependencies using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn 
pip install re 
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Krinal09/Fake-News-Detection.git
```

2. Navigate to the project directory:

```bash
cd fake-news-detection
```

3. Execute the Jupyter Notebook or Python scripts associated with each classifier to train and test the models. For example:

```bash
python random_forest_classifier.py
```

4. The code will produce evaluation metrics and provide a prediction for whether the given news is true or false based on the trained model.

## Results

We evaluated each classifier's performance using metrics such as accuracy, precision, recall, and F1 score. The results are documented in the project files.

## Model Deployment

# Mandatory Conditional Logic Gate (Model Promotion)

To ensure reliable deployment, a conditional logic gate is implemented in the pipeline.

Deployment Rule:

The improved model is promoted only if its F1-score is greater than or equal to the production baseline model.

```bash
BASELINE_F1_THRESHOLD = 0.94

if new_model_f1 >= BASELINE_F1_THRESHOLD:
    print("New model approved for deployment")
    deploy_model = True
else:
    print("New model rejected – baseline performs better")
    deploy_model = False
```

This logic prevents performance regression and enforces production-level model governance.

# MLflow UI

To visualize experiment runs locally:

```bash
mlflow ui
```

Then open the browser at:

```bash
http://127.0.0.1:5000
```

