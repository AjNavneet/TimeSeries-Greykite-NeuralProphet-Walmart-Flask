# Time Series Forecasting with Greykite, Neural Prophet, and Flask Deployment

## Business Overview

Time series data, collected at regular intervals, is crucial for businesses to understand past trends and forecast future outcomes. Accurate time series forecasting enables organizations to make data-driven decisions and optimize operations across various domains, including supply chain management, financial forecasting, weather prediction, and sales analysis.

This project utilizes cutting-edge tools such as **Greykite**, a Python library developed by LinkedIn, and **Neural Prophet**, a deep-learning-based model developed by Facebook, to predict store sales. The solution also integrates **Flask** for web deployment, making forecasts easily accessible to end users.

---

## Aim

The goal is to develop an accurate time series forecasting solution that leverages historical data to predict future sales trends using **Greykite** and **Neural Prophet**, with an accessible deployment interface via Flask.

---

## Dataset Description

This project uses Walmart store sales data, which contains detailed sales records for 45 Walmart stores in different regions. The dataset includes:

1. **Stores.csv**: Information about each store, including store type and size.
2. **Train.csv**: Historical sales data spanning from 2010-02-05 to 2012-11-01.
3. **Test.csv**: Similar to Train.csv but excludes weekly sales, which need to be predicted.
4. **Features.csv**: Supplementary data related to stores, departments, and regional activities.

### Key Features:
- **Store**: Unique identifier for stores.
- **Date**: Weekly sales data timestamps.
- **Department**: Identifier for store departments.
- **Temperature**: Average regional temperature.
- **Fuel Price**: Weekly fuel prices.
- **MarkDown1-5**: Anonymized promotional markdown data.
- **CPI**: Consumer Price Index.
- **Unemployment**: Regional unemployment rate.
- **IsHoliday**: Indicator for special holiday weeks.
- **Weekly Sales**: Target variable for predictions.

---

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `Greykite` and `Neural Prophet` for advanced time series forecasting.
  - `Pandas` and `NumPy` for efficient data handling.
  - `Matplotlib`, `Seaborn`, and `Plotly` for interactive visualizations.
  - `Pandas-Profiling` for automated exploratory data analysis.
  - `Sci-kit Learn` for preprocessing and metrics.
  - `Flask` for web-based deployment.

---

## Approach

### 1. Exploratory Data Analysis (EDA):
- Analyze dataset features using `Pandas-Profiling`.
- Visualize data trends, seasonality, and anomalies.

### 2. Data Cleaning:
- Handle missing data and outliers.
- Normalize numerical features and encode categorical variables.

### 3. Feature Engineering:
- Extract date components (day, month, year, week).
- Engineer features based on domain-specific knowledge.

### 4. Time Series Component Analysis:
- Decompose time series into trend, seasonality, and residuals.

### 5. Model Development:
- Train forecasting models with:
  - **Greykite**: Provides interpretable and scalable forecasting solutions.
  - **Neural Prophet**: Combines deep learning with time series methodologies.

### 6. Model Evaluation:
- Assess model performance using metrics:
  - Mean Absolute Percent Error (MAPE)
  - Root Mean Squared Error (RMSE)

### 7. Forecasting:
- Generate weekly sales forecasts for departments and stores.

### 8. Deployment:
- Develop a Flask-based web application for real-time predictions.

---

## Project Structure

```plaintext
.
├── input/                                # Data files (stores.csv, train.csv, test.csv, features.csv).
├── src/                                  # Core project folder.
│   ├── engine.py                         # Main script for model training and evaluation.
│   ├── server.py                         # Flask app for deployment.
│   ├── ML_Pipeline/                      # Modular scripts for preprocessing and modeling.
│       ├── data_processing.py            # Data cleaning utilities.
│       ├── forecasting_models.py         # Greykite and Neural Prophet implementations.
│       ├── feature_engineering.py        # Feature extraction and engineering.
├── output/                               # Results and saved models.
├── lib/                                  # Reference materials and notebooks.
├── requirements.txt                      # List of dependencies.
└── README.md                             # Project documentation.
```

---

## Getting Started

### 1. Clone the Repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies:

Install the required libraries with:

```bash
pip install -r requirements.txt
```

### 3. Run the Project:

- Train models and generate forecasts:

```bash
python src/engine.py
```

- Deploy the Flask application:

```bash
python src/server.py
```

### 4. Access the Application:

Navigate to `http://127.0.0.1:5000` in your browser to interact with the web-based forecasting tool.

---

## Results

- **Accurate Forecasts**:
  - Models effectively predict sales trends and seasonality.
- **Interactive Visualizations**:
  - Visualizations provide actionable insights for business decision-making.
- **Scalable Deployment**:
  - Flask application ensures easy integration into business workflows.

---

## Why Choose This Project?

- **Advanced Models**: Leverage state-of-the-art tools like Greykite and Neural Prophet.
- **Comprehensive Workflow**: From data preprocessing to web deployment.
- **Real-World Applications**: Designed to solve practical business challenges in forecasting.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature"
```

4. Push your branch:

```bash
git push origin feature-name
```

5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or suggestions, please reach out to:

- **Name**: Abhinav Navneet
- **Email**: mailme.AbhinavN@gmail.com
- **GitHub**: [AjNavneet](https://github.com/AjNavneet)

---

## Acknowledgments

Special thanks to:

- [Greykite](https://linkedin.github.io/greykite/) for its powerful forecasting library.
- [Neural Prophet](https://facebook.github.io/neuralprophet/) for deep-learning forecasting capabilities.
- The Python open-source community for exceptional tools and support.

---
