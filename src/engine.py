# Importing required packages
import datetime
import warnings
from ML_pipeline import Deploy, Preprocess, Train_Model, Utils
from greykite.framework.templates.autogen.forecast_config import MetadataParam

# Suppressing all the warnings
warnings.filterwarnings("ignore")

# Input 0 to train the model and 1 to deploy the model
val = int(input("Train - 0\nDeploy - 1\nEnter your value: "))
if val == 0:
    # Loading the data
    train = Utils.read_data('../input/train.csv').drop_duplicates()
    feature = Utils.read_data('../input/features.csv')
    stores = Utils.read_data('../input/stores.csv')
    test = Utils.read_data('../input/test.csv')
    print("Data is loaded")

    print("Data preprocessing started")

    # Merging the data frames
    train_m1 = Utils.merge_dataframes(train, stores)
    train_data = Utils.merge_dataframes(train_m1, feature)

    agg_train_col = {"Weekly_Sales": sum, "IsHoliday": "first",
                    "Type": "first", "Size": "first", "Temperature": "first",
                    "Fuel_Price": "first", "MarkDown1": "first", "MarkDown2": "first",
                    "MarkDown3": "first", "MarkDown4": "first", "MarkDown5": "first",
                    "CPI": "first", "Unemployment": "first"}

    agg_test_col = {"IsHoliday": "first",
                    "Type": "first", "Size": "first", "Temperature": "first",
                    "Fuel_Price": "first", "MarkDown1": "first", "MarkDown2": "first",
                    "MarkDown3": "first", "MarkDown4": "first", "MarkDown5": "first",
                    "CPI": "first", "Unemployment": "first"}

    # Grouping the dataframe by date
    train_data = Preprocess.group_data(train_data, "Date", agg_train_col)

    # Imputing missing value by 0
    train_data = Preprocess.impute(train_data)

    # Replacing the outliers in the target variable
    train_data.Weekly_Sales = Preprocess.replace_outliers(train_data.Weekly_Sales, 2000000, 2000000)

    # Adding new columns in data for year, month, and day
    new_col = ['Date_year', 'Date_month', 'Date_day', 'Date_dayofweek']
    date_col = 'Date'
    train_data = Preprocess.separate_date_col(train_data, date_col, new_col)

    # Mapping
    type_mapping = {"A": 1, "B": 2, "C": 3}
    train_data = Preprocess.map(train_data, 'Type', type_mapping)

    holiday_type_mapping = {False: 0, True: 1}
    train_data = Preprocess.map(train_data, 'IsHoliday', holiday_type_mapping)

    # Dropping the features
    features_drop = ['Unemployment', 'CPI', 'MarkDown5']
    train_data = Preprocess.drop_col(train_data, features_drop)

    # Changing the type of date column
    train_data = Preprocess.change_type(train_data, 'Date', 'datetime64[ns]')

    # Renaming the columns
    rename_col = {'Date': 'ds', 'Weekly_Sales': 'y'}
    train_data = Preprocess.rename_column(train_data, rename_col)

    # Selecting specific features in the dataset
    select_col = ['ds', 'Temperature', 'Fuel_Price', 'IsHoliday', 'y']
    final_train_data = Preprocess.select_features(train_data, select_col)

    # Sorting the data for model training
    final_data = Preprocess.sort_data(final_train_data, 'ds')
    print("Data preprocessing ended")

    # Model Training
    print("Model training has started")

    # Silverkite model training
    metadata = MetadataParam(
        time_col="ds",  # name of the time column
        value_col="y",  # name of the value column
        freq="W-FRI",  # "MS" for Monthly at start date, "H" for hourly, "D" for daily, "W" for weekly, etc.
        train_end_date=datetime.datetime(2012, 6, 8)
    )
    regressor_col = {"regressor_cols": ["Temperature", "Fuel_Price", "IsHoliday"]}

    greykite_model = Train_Model.train_greykite(final_data, metadata, regressor_col)

    # Neural prophet model training
    future_regressors = ['Temperature', 'Fuel_Price', 'IsHoliday']
    prophet_model = Train_Model.train_neural_prophet(final_data, future_regressors)
    p_path = '../output/prophet_model.pkl'
    Utils.save_model(prophet_model, p_path)
    print('Neural Prophet model is saved as a pkl file in ' + str('../output/prophet_model.pkl'))

else:
    # Deploying the model
    p_path = '../output/prophet_model.pkl'
    Deploy.init(p_path)
