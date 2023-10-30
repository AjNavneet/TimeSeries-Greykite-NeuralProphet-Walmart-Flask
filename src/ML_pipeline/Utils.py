# Importing required packages
import pickle
import pandas as pd

# Function to save a machine learning model to a file
def save_model(model, path):
    with open(path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

# Function to load a machine learning model from a file
def load_model(path):
    with open(path, 'rb') as model:
        ml_model = pickle.load(model)
        return ml_model

# Function to read data from a file using pandas
def read_data(file_path, **kwargs):
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data

# Function to merge two data frames using pandas
def merge_dataframes(df1, df2):
    combined_data = pd.merge(df1, df2)
    return combined_data
