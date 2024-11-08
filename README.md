# Autism Spectrum Disorder Prediction Using Machine Learning

## Overview
This project aims to predict the risk of Autism Spectrum Disorder (ASD) using machine learning models. It involves multiple stages, including data preprocessing, model training, hyperparameter tuning, and deployment of the final model as a Streamlit application. The goal is to help identify potential signs of autism in children based on a questionnaire.

## Project Files
- **app.py**: The Streamlit application that allows users to input answers to a questionnaire and get a risk prediction for autism. The model used in this application is pre-trained and saved as `trained_model.pkl`.
- **main.py**: The main script that includes data preprocessing, training, evaluation, and hyperparameter tuning of various machine learning models.
- **sql_questions.sql**: Contains SQL queries used to perform data analysis on various aspects of the dataset, such as gender distribution and ethnicity among individuals with autism.
- **presentation.pdf**: Presentation slides that summarize the project, including the overview of ASD, the models used, the average scores achieved, and future directions.
- **tableau.twb**: Tableau workbook used to visualize and analyze data related to ASD screening.

## Machine Learning Models
The models used in this project include:
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Naive Bayes (GaussianNB and MultinomialNB)**
- **MLPClassifier (Neural Network)**
- **SGDClassifier (Stochastic Gradient Descent)**
- **KNeighborsClassifier (K-Nearest Neighbors)**
- **Decision Tree Classifier**
- **Random Forest Classifier** (with hyperparameter tuning)
- **Gradient Boosting Classifier** (with hyperparameter tuning)
- **LGBMClassifier**
- **XGBoost Classifier**

The models were evaluated using cross-validation, and the average ROC AUC score achieved was 0.9091, indicating that the model is effective in correctly identifying classes with over 90% accuracy.

## Streamlit Application
The Streamlit app (`app.py`) allows users to interact with the model by answering questions related to social behavior, communication, and routine changes. Based on these answers, the app predicts the risk of autism and provides guidance on the next steps for assessment.

To run the app, use the following command:
```bash
streamlit run app.py
```
Make sure to have the pre-trained model (`trained_model.pkl`) in the `code` directory before running the app.

## SQL Queries
The project also includes analysis performed using SQL, such as:
1. The number of users diagnosed with autism.
2. Gender distribution among individuals with autism.
3. The average age of individuals with autism versus those without.
4. Main ethnicities among individuals diagnosed with autism.
5. The link between family history of autism and diagnosis of ASD.

These queries help provide more insights into the dataset and support data-driven decision-making.

## Data Visualization
The **Tableau** workbook (`tableau.twb`) contains visualizations that provide insights into the dataset, such as age distribution, gender distribution, and other factors related to autism diagnosis.

## Presentation
The **presentation.pdf** file provides a summary of the project, including an introduction to ASD, the machine learning models used, the results achieved, and future steps. It serves as a comprehensive overview of the project's objectives and outcomes.

## How to Use
1. Clone the repository to your local machine.
2. Ensure you have Python and the necessary libraries installed.
3. Run the Streamlit app (`app.py`) to interact with the prediction model.
4. Use the provided SQL queries to further analyze the dataset.
5. Explore the **Tableau** workbook for data visualizations.

## Requirements
- Python 3.7+
- Streamlit
- Joblib
- Pandas
- Numpy
- Scikit-learn
- XGBoost
- LightGBM
- Tableau (for visualization)

## Future Directions
The project can be further enhanced by:
- Collecting more diverse datasets to improve model generalization.
- Incorporating additional features such as genetic or medical history for better predictions.
- Deploying the model as a web service for broader accessibility.

## References
- [What is Autism? | Cincinnati Children's](https://www.youtube.com/watch?v=hwaaphuStxY)
- [Autistic Spectrum Disorder Screening Data for Children](https://archive.ics.uci.edu/dataset/419/autistic%2Bspectrum%2Bdisorder%2Bscreening%2Bdata%2Bfor%2Bchildren)
- [Speeding Autism Diagnosis, Improving Outcomes Using Machine Learning](https://today.duke.edu/2019/07/speeding-autism-diagnosis-improving-outcomes-using-machine-learning)
- [APPDA Lisbon](https://appda-lisboa.org.pt/autismo/diagnostico)
- [Diagnostic Approach and Intervention in Autism Spectrum Disorder in Pediatric and Adult Age (DGS - Portugal)](https://normas.dgs.min-saude.pt/wp-content/uploads/2019/09/Abordagem-Diagnostica-e-Intervencao-na-Perturbacao-do-Espetro-do-Autismo-em-Idade-Pediatrica-e-no-Adulto-2019.pdf)

## License
This project is licensed under the MIT License.
