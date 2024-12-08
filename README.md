# Fake News Detection

This project is a machine learning application to detect fake news articles using a dataset from Kaggle. The dataset contains news articles labeled as reliable (`0`) or unreliable (`1`), and the model predicts whether a given article is fake or real.

## Dataset

The dataset used in this project is provided by the Kaggle competition on Fake News detection. You can find the dataset at the following link:  
[Fake News Dataset](https://www.kaggle.com/c/fake-news%20/data)

### Dataset Description
- **train.csv**: Training dataset with the following attributes:
  - `id`: Unique ID for a news article.
  - `title`: Title of the news article.
  - `author`: Author of the news article.
  - `text`: The body of the news article (can be incomplete).
  - `label`: Target variable (1 = unreliable, 0 = reliable).

## Features of the Project

1. **Data Preprocessing**:
   - Cleaning text data (removing special characters, HTML tags, URLs, and stopwords).
   - Combining `title` and `text` for feature extraction.

2. **Feature Extraction**:
   - Using TF-IDF vectorization to convert text into numerical features.

3. **Model Training**:
   - Training and evaluating multiple machine learning models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - Gradient Boosting
   - Comparison of model performance using accuracy scores.

4. **Visualization**:
   - Word cloud for data exploration.
   - Confusion matrix to evaluate model predictions.
   - Bar graphs to compare model performance.

5. **Deployment**:
   - A modular pipeline for training and testing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
