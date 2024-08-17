# import libraries
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



# Function to load data (5 pts)
def load_data(file_path):
    # Load data from the CSV file or another format and return data
    data =  pd.read_csv(file_path)
    return data
    

# Function to preprocess data (handling missing and outlier data) (15 pts)
def preprocess_data(df):
    # Handle missing data using appropriate imputation
    for colname in df.columns: 
        missed_values = df.isnull().sum().sum()
        if(missed_values > 0):
           df.dropna(inplace = True)
    # Deal with outlier data 

        q1 = np.percentile(df[colname], 25)
        q3 = np.percentile(df[colname], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr

        df.loc[(df[colname] < lower_bound), colname] = lower_bound
        df.loc[(df[colname] > upper_bound), colname] = upper_bound
    # return data

    # return data
    return df

# Function to split data into training and testing sets (5 pts)
def split_data(df): 
    # Split data into training (80%) and testing (20%) sets
    X = df.drop('Outcome', axis=1)
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test 

# Function to train a model with hyperparameters (30 pts)
def train_model(X_train, y_train): 
    # Train a or many models with hyperparameter tuning

    param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
      }

    nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid= param_grid_nb, verbose=1, cv=10, n_jobs=-1)
    nbModel_grid.fit(X_train, y_train)
    

    return nbModel_grid.best_estimator_
    
   

# Function to evaluate the model (15 pts)
def evaluate_model(model, X_test, y_test):
     # Make prediction of the model
    y_pred = model.predict(X_test)

    # Evaluate the best model
    print(confusion_matrix(y_test, y_pred), ": La matrice de confusion est:")
    print(accuracy_score(y_test, y_pred), ": Le score de prediction est:")
    print(precision_score(y_test, y_pred), ": Le score de precision est: ")
    print(recall_score(y_test, y_pred), ": Le recall score est :")
    print(f1_score(y_test, y_pred), ": Le f1 score est :")
    # Evaluate the best model 
    pass

# Function to deploy the model (bonus) (10 pts)
def deploy_model(model, X_test):
    # Deploy the best model using Streamlit or Flask (bonus)
    pass

# Main function
def main():
    # Load data
    data = load_data("diabetes.csv")
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    
    # Train a model with hyperparameters
    best_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
    
    # Deploy the model (bonus)
    # deploy_model(best_model, X_test)

if __name__ == "__main__":
    main()

