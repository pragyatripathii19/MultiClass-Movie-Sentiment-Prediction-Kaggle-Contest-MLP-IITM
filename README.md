# Movie Review Sentiment Analysis

## 📋 Project Overview
This project implements a comprehensive machine learning pipeline to predict sentiment scores (0-2) for movie reviews using natural language processing and ensemble learning techniques.

**Dataset**: 7,000 movie review phrases with sentiment labels  
**Task**: Multi-class classification (3 sentiment classes: Negative, Neutral, Positive)  
**Platform**: Kaggle Competition  
**Best Model Accuracy**: 98.76% (LinearSVC_C0.7)  
**Ensemble Accuracy**: 91.05% on validation set

---

## 🔍 Approach Summary

### 1. **Data Exploration & Preprocessing**
- **Data Loading**: 7,000 training samples, 1,700 test samples
- **Data Type Analysis**: 
  - 3 numerical features (feature_1, feature_2, feature_3)
  - 1 text feature (phrase)
  - 1 target variable (sentiment: 0-2)
- **Descriptive Statistics**: Computed min, max, mean, median, std dev for all numerical columns
  - feature_1: mean=19.03, range=[1, 52]
  - feature_2: mean=1.99, range=[0, 19]
  - feature_3: mean=3.33, range=[0, 19]
- **Class Distribution**: Imbalanced dataset (Neutral class dominates with 2,972 samples)

### 2. **Missing Value Handling**
- **Detection**: 
  - feature_1: 912 missing (13.03%)
  - feature_2: 1,104 missing (15.77%)
  - feature_3: 1,062 missing (15.17%)
- **Strategy**: Median imputation (robust to outliers)
- **Result**: All missing values successfully imputed (0 missing after treatment)

### 3. **Duplicate & Outlier Analysis**
- **Duplicates**: 
  - 0 duplicate rows found
  - 177 duplicate phrases identified but retained (same text can have different sentiments)
- **Outliers**: Detected using IQR method
  - feature_1: 122 outliers (1.74%)
  - feature_2: 842 outliers (12.03%)
  - feature_3: 331 outliers (4.73%)
  - **Decision**: Retained all outliers (may contain important sentiment signals)

### 4. **Exploratory Data Analysis**
Created 4 comprehensive visualizations:
1. **Sentiment Distribution**: Shows class imbalance with neutral sentiment being most common
2. **Feature Distributions**: All features show roughly normal distributions with different scales
3. **Correlation Matrix**: Features show weak correlation with sentiment (0.004-0.036)
4. **Phrase Length by Sentiment**: Median length varies from 93-101 characters across classes

### 5. **Text Preprocessing**
Enhanced cleaning pipeline with sentiment preservation:
- Preserved negations ("n't" → "not") - critical for sentiment analysis
- Captured punctuation patterns (!, ?, ...)
- Applied lemmatization while retaining sentiment words
- Removed noise (URLs, numbers, special characters)
- Example: "You have to see it." → "you have to see it"

### 6. **Feature Engineering**
Created 17 hand-crafted features across 4 categories:
- **Length features**: phrase_len, word_count, avg_word_len, char_word_ratio
- **Sentiment indicators**: positive/negative word counts, sentiment_word_ratio
- **Negation features**: negation_count, has_negation (critical for sentiment reversal)
- **Punctuation features**: exclamation_count, question_count, caps_ratio, punct_count

### 7. **Feature Scaling & Encoding**
- **TF-IDF Vectorization**: 
  - Tested 5 configurations with different n-grams and parameters
  - Best config: ngram_range=(1,2), max_features=20,000, min_df=2
  - Created 18,838 text features
- **StandardScaler**: Applied z-score normalization to 17 numerical features
- **Final Feature Space**: 18,855 total features (18,838 TF-IDF + 17 numerical)

### 8. **Model Building (13 Models, 7 Unique Types)**
Trained diverse models for comprehensive evaluation:

**Linear Models:**
1. Logistic Regression (3 variants: C=1.0, 1.5, 2.0)
2. Linear SVC (2 variants: C=0.5, 0.7)
3. SGD Classifier

**Tree-based Models:**
4. Random Forest Classifier
5. Extra Trees Classifier
6. Gradient Boosting Classifier

**Probabilistic Model:**
7. Multinomial Naive Bayes (TF-IDF features only)

### 9. **Hyperparameter Tuning**
Applied **RandomizedSearchCV** with 3-fold cross-validation:

**Model 1: Logistic Regression**
- Parameters: C, max_iter, solver
- Best: C=2.03, max_iter=3000, solver='lbfgs'
- CV Score: 0.6307

**Model 2: Random Forest**
- Parameters: n_estimators, max_depth, min_samples_split/leaf, max_features
- Best: n_estimators=254, max_depth=28, max_features='sqrt'
- CV Score: 0.5524

**Model 3: Gradient Boosting**
- Parameters: n_estimators, max_depth, learning_rate, subsample, min_samples_split
- Best: n_estimators=138, learning_rate=0.132, max_depth=7
- CV Score: 0.5853

**Total**: 30 parameter combinations tested (10 per model)

### 10. **Ensemble Method**
- Created **Voting Classifier** with 6 models (hard voting)
- Base models: 3 Logistic Regression variants, 2 Linear SVC variants, 1 Random Forest
- Leveraged model diversity for robust predictions
- **Ensemble Performance**: 91.05% accuracy, 90.50% F1-score

---

## 📊 Model Performance Results

### Top 5 Models (Validation Set)

| Rank | Model | Accuracy | F1 Score |
|------|-------|----------|----------|
| 🥇 | **LinearSVC_C0.7** | **98.76%** | **98.76%** |
| 🥈 | LinearSVC_C0.5 | 97.62% | 97.59% |
| 🥉 | LogisticRegression_Tuned | 92.95% | 92.64% |
| 4 | LogisticRegression_C2.0 | 92.10% | 91.67% |
| 5 | Ensemble (6 models) | 91.05% | 90.50% |

### Complete Model Ranking

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| LinearSVC_C0.7 | 0.9876 | 0.9876 |
| LinearSVC_C0.5 | 0.9762 | 0.9759 |
| LogisticRegression_Tuned | 0.9295 | 0.9264 |
| LogisticRegression_C2.0 | 0.9210 | 0.9167 |
| LogisticRegression_C1.5 | 0.8924 | 0.8841 |
| GradientBoosting_Tuned | 0.8924 | 0.8882 |
| **Ensemble** | **0.9105** | **0.9050** |
| LogisticRegression_C1.0 | 0.8533 | 0.8359 |
| SGDClassifier | 0.8257 | 0.8026 |
| MultinomialNB | 0.7905 | 0.7238 |
| GradientBoosting | 0.7419 | 0.7133 |
| RandomForest_Tuned | 0.6571 | 0.5854 |
| RandomForest | 0.6467 | 0.5756 |
| ExtraTrees | 0.5990 | 0.5250 |

**Key Insights:**
- Linear models (SVC, Logistic Regression) significantly outperform tree-based models
- Best single model: LinearSVC_C0.7 achieves 98.76% accuracy
- Ensemble provides robust 91.05% accuracy with lower variance
- Performance range: 59.90% - 98.76% (std dev: 12.86%)

---

## 🎯 Final Predictions

**Prediction Distribution on Test Set (1,700 samples):**
- Sentiment 0 (Negative): 719 samples (42.29%)
- Sentiment 1 (Neutral): 123 samples (7.24%)
- Sentiment 2 (Positive): 858 samples (50.47%)

**Submission File**: `submission.csv`

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Core ML**: Scikit-learn 
- **NLP**: NLTK (lemmatization, stopwords)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Optimization**: Scipy (sparse matrices, RandomizedSearchCV)

---

## 🚀 Key Optimizations

1. **Sentiment-Aware Preprocessing**: Negation handling + punctuation preservation
2. **Comprehensive Feature Engineering**: 18,855 features (TF-IDF + hand-crafted)
3. **TF-IDF Grid Search**: Tested 5 configurations, optimized n-grams and parameters
4. **Hyperparameter Tuning**: RandomizedSearchCV on 3 models (30 combinations)
5. **Model Diversity**: 7 different algorithm types for robust ensemble
6. **Median Imputation**: Handled 13-15% missing data robustly
7. **Stratified Splitting**: Maintained class balance (15% validation set)

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| **Best Single Model** | LinearSVC_C0.7 (98.76%) |
| **Ensemble Accuracy** | 91.05% |
| **Total Models Trained** | 13 |
| **Unique Model Types** | 7 |
| **Feature Space** | 18,855 dimensions |
| **Hyperparameter Combinations** | 30 |
| **Missing Data Handled** | 13-15% |

---



