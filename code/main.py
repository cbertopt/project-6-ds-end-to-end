# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import joblib

# Step 2: Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Step 3: Load the dataset
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
sample_df = pd.read_csv('data/sample_submission.csv')

# Function to preprocess datasets
def preprocess_data(train_df, test_df):
    # Step 4: Identify the target column
    target_col = [x for x in train_df.columns if x not in test_df.columns][0]

    # Step 5: Replace special characters in specific columns
    for col in ['relation', 'ethnicity']:
        train_df[col] = train_df[col].replace('?', 'Others')
        test_df[col] = test_df[col].replace('?', 'Others')

    # Step 6: Drop 'age_desc' column from both datasets
    for col in ['age_desc', 'used_app_before', 'gender']:
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)

    # Step 7: One-hot encoding for categorical variables with only 2 unique values
    for col in ['jaundice', 'austim']:
        train_df[col] = np.where(train_df[col] == 'yes', 1, 0)
        test_df[col] = np.where(test_df[col] == 'yes', 1, 0)

    # Step 8: Label encoding for categorical variables
    for col in ['ethnicity', 'contry_of_res', 'relation']:
        categories = train_df[col].value_counts().index
        mapping = dict(zip(categories, range(1, len(categories) + 1)))
        train_df[col] = train_df[col].map(mapping)
        test_df[col] = test_df[col].map(mapping)

    # Step 9: Fill missing values using SimpleImputer with the most frequent value (mode)
    imputer = SimpleImputer(strategy='most_frequent')
    test_df['contry_of_res'] = imputer.fit_transform(test_df[['contry_of_res']])

    return train_df, test_df, target_col

# Step 10: Preprocess the datasets
train_df, test_df, target_col = preprocess_data(train_df, test_df)

# Step 11: Prepare training features and target variable
X = train_df.drop(['ID', target_col], axis=1)
y = train_df[target_col]

# Step 12: Define classifiers
seed = 123
model1 = LogisticRegression(max_iter=500, random_state=seed)
model2 = SVC(random_state=seed)
model3 = GaussianNB()
model4 = MLPClassifier(random_state=seed, max_iter=500)
model5 = SGDClassifier(random_state=seed)
model6 = KNeighborsClassifier()
model7 = DecisionTreeClassifier(random_state=seed)
model8 = RandomForestClassifier(random_state=seed, class_weight="balanced")
model9 = GradientBoostingClassifier(random_state=seed)
model10 = LGBMClassifier(random_state=seed)
model11 = XGBClassifier(random_state=seed, use_label_encoder=False)

# Step 13: Hyperparameter tuning for RandomForest and GradientBoosting
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}

rf_grid = GridSearchCV(model8, param_grid_rf, cv=3, scoring='roc_auc', verbose=1)
gb_grid = GridSearchCV(model9, param_grid_gb, cv=3, scoring='roc_auc', verbose=1)

rf_grid.fit(X, y)
gb_grid.fit(X, y)

# Step 14: Use best estimators from GridSearchCV
model8 = rf_grid.best_estimator_
model9 = gb_grid.best_estimator_

# Step 15: Evaluate models with cross-validation and print scores
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]

for idx, model in enumerate(models):
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
    print(f"Model {idx + 1} - Mean AUC: {np.mean(scores):.4f}")

# Step 16: Feature importance analysis for tree-based models
feature_importances = model8.feature_importances_
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print(feature_df)

# Step 17: Create weighted ensemble with top 3 algorithms
kfold = 5
skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
X1 = np.array(X)
y1 = np.array(y)
scores = []

# Step 17.1: Perform Stratified K-Fold cross-validation
for i, (train_index, test_index) in enumerate(skf.split(X1, y1)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X1[train_index], X1[test_index]
    y_train, y_valid = y1[train_index], y1[test_index]
    
    # Train top 3 models
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict_proba(X_valid)[:, 1]
    
    model8.fit(X_train, y_train)
    y_pred2 = model8.predict_proba(X_valid)[:, 1]
    
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict_proba(X_valid)[:, 1]
    
    # Step 17.2: Calculate weighted average prediction and ROC AUC score
    y_pred = 0.2 * y_pred1 + 0.15 * y_pred2 + 0.65 * y_pred3
    score = roc_auc_score(y_valid, y_pred)
    scores.append(score)
    print(f'Fold {i + 1}/{kfold} - Score: {score:.4f}')

# Step 18: Calculate and print average score across all folds
print(f"Avg scores - {np.mean(scores):.4f}")

# Step 19: Saving the trained model
joblib.dump(model, 'trained_model.pkl')