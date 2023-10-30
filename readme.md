# Time Series Forecasting with Greykite, Neural Prophet and Flask Deployment

## Business Overview

Time series data, collected at regular time intervals, is essential for businesses to understand how past events influence the future. Forecasting is the process of estimating future observations based on historical data. Time series forecasting is a statistical method used to analyze time-based patterns in data, helping organizations model and predict future behavior. It serves as a bridge connecting the past, present, and future.

Forecasting is vital in various domains, including supply chain management, stock prediction, weather forecasting, and biomedical monitoring. In this project, we aim to predict store sales using Greykite, a Python library developed by LinkedIn, and the Neural Prophet model developed by Facebook.

---

## Aim

The objective is to predict future sales/demand using historical data and other relevant features using Greykite and Neural Prophet.

---

## Dataset Description

We use Walmart store sales data, which includes historical sales data for 45 Walmart stores located in different regions. Each store contains multiple departments. The dataset comprises four main files:

1. **Stores.csv:** Information about the 45 stores, including their type and size.
2. **Train.csv:** Historical training data covering the period from 2010-02-05 to 2012-11-01.
3. **Test.csv:** Identical to train.csv, except it lacks the weekly sales that need to be predicted.
4. **Features.csv:** Additional data related to stores, departments, and regional activities for specific dates.

The key features in the dataset include:
- Store number
- Date (week)
- Department number
- Average temperature in the region
- Fuel price
- MarkDown1-5 (anonymized data related to promotional markdowns)
- Consumer price index (CPI)
- Unemployment rate
- Special holiday weeks (IsHoliday)
- Weekly sales for a given department in a store

---

## Tech Stack

- **Language:** `Python`
- **Libraries:** `Greykite`, `Neural Prophet`, `Sci-kit Learn`, `Pandas`, `Pandas Profiling`, `Matplotlib`, `Datetime`, `Plotly`, `Seaborn`, `Numpy`

---

## Approach

1. **Exploratory Data Analysis (EDA):**
   - Feature analysis
   - Data visualization using Pandas Profiling

2. **Data Cleaning:**
   - Handling missing values
   - Detecting and handling outliers

3. **Feature Engineering:**
   - Extracting day, month, and year from the date
   - Mapping and encoding

4. **Time Series Component Analysis:**
   - Analyzing trends and seasonality

5. **Model Building:**
   - Greykite
   - Neural Prophet

6. **Model Evaluation:**
   - Mean Absolute Percent Error
   - RMSE

7. **Forecasting Using Trained Models**

---

## Modular Code Overview

1. **Input:** Contains the data files used for analysis (features.csv, stores.csv, test.csv, and train.csv).

2. **Src:** The core of the project, containing modularized code for various steps:
   - ML_pipeline
   - engine.py
   - server.py

3. **Output:** Contains trained models for future use.

4. **Lib:** Reference materials, including the original IPython notebook

5. **requirements.txt:** Lists all required libraries and their versions. Install these libraries using `pip install -r requirements.txt`.

Note: For installing the Neural Prophet and Greykite libraries, refer to the document "Steps to Install Neural Prophet and Greykite Libraries."

---

## Project Takeaways

This project provides insights into a wide range of topics, including:

1. Understanding the business context and objectives.
2. Analyzing trends and seasonality.
3. Using Greykite for time series modeling.
4. Employing the Neural Prophet library for forecasting.
5. Model evaluation methods.
6. Comparing Greykite and Neural Prophet models.
7. Deployment using Flask.

---