# 🍽️ Restaurant Sales Analytics & Demand Forecasting

A machine learning project that analyzes restaurant sales, forecasts daily demand for each menu item, and converts those forecasts into practical food-preparation recommendations.

## 1. Project Overview

Restaurants need to prepare enough food to satisfy customer demand without producing excessive leftovers. Poor demand estimates can cause:

- **Food waste:** preparing more food than customers purchase.
- **Stockouts:** running out of popular menu items and potentially losing sales.
- **Inefficient staffing:** failing to recognize busy days and peak hours.

This project uses historical sales patterns and machine learning to estimate demand, evaluate forecast accuracy, and recommend preparation quantities with configurable safety buffers.

> **Dataset note:** The notebook defaults to a deliberately messy synthetic dataset. It simulates restaurant transactions; it is not real restaurant customer data. The notebook can also be configured to load a user's own CSV file.

## 2. Main Features

- Generate approximately two years of synthetic restaurant transactions.
- Optionally load a custom CSV dataset.
- Clean missing values, duplicate rows, returns, inconsistent item names, and extreme quantity outliers.
- Analyze daily sales, popular menu items, and peak selling hours.
- Build historical and calendar-based forecasting features.
- Train a HistGradientBoostingRegressor using a Poisson loss.
- Evaluate the model against simple forecasting baselines.
- Simulate the trade-off between food waste and shortages at different service levels.
- Generate a demand forecast and preparation plan for the next date in the dataset.
- Analyze feature importance using permutation importance.
- Export model outputs and a trained model.
- Provide an interactive Streamlit dashboard.

## 3. Project Workflow

1. **Data generation or loading** — create synthetic sales transactions or load a custom CSV.
2. **Data cleaning** — standardize and filter the transaction records.
3. **Exploratory data analysis** — inspect sales trends, top-selling products, and busy hours.
4. **Feature engineering** — create lag, rolling-statistic, weekday, calendar, and item features.
5. **Model training** — fit a gradient boosting regression model using a chronological split.
6. **Evaluation** — compare forecasts with actual sales and baseline methods.
7. **Preparation planning** — add a statistical safety buffer to the expected demand.
8. **Explainability** — estimate which features influence predictive performance.
9. **Export and dashboard** — save outputs and display them through Streamlit.

## 4. Technologies Used

| Technology | Role |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations and synthetic data generation |
| Pandas | Data manipulation and aggregation |
| Matplotlib | Exploratory plots and model diagnostics |
| Scikit-learn | Gradient boosting model and permutation importance |
| SciPy | Normal distribution quantiles for service-level buffers |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| Plotly | Interactive dashboard charts |

## 5. Dataset

### Default synthetic dataset

The notebook generates approximately 730 days of transaction data using a fixed random seed (`42`) for reproducibility.

The synthetic menu contains 12 items, including pizzas, burgers, salads, chicken, fries, desserts, and drinks.

The simulation includes:

- Different average demand levels by menu item.
- Different lunch and dinner sales profiles.
- Day-of-week demand multipliers.
- Seasonal variation and gradual growth.
- Random daily variation.
- Occasional item promotions.
- A simulated Christmas closure.
- Deliberately introduced missing quantities, returns, duplicate rows, inconsistent item capitalization, and extreme quantity values.

### Using your own dataset

Set:

```python
USE_SYNTHETIC = False
```

The notebook then prompts for a CSV upload in Google Colab.

Configure the source column names in `COLS`:

```python
COLS = dict(
    datetime="Date",
    item="Item",
    qty="Quantity",
    price="Price"
)
```

The required fields are date/time, item name, and quantity. Price is optional.

For custom data, the notebook expects transaction-level sales lines that can be mapped to these fields.

## 6. Data Cleaning and Preprocessing

The `clean_sales()` function performs the following operations:

1. Renames source columns to standard internal names.
2. Parses date/time values and converts quantities to numeric values.
3. Removes rows with missing date/time, item, or quantity.
4. Trims whitespace, normalizes repeated spaces, and standardizes item capitalization.
5. Removes exact duplicate rows.
6. Removes zero and negative quantities, including returns.
7. Removes extreme quantity outliers using an item-specific threshold based on the 99.9th percentile.
8. Removes items with fewer than 60 distinct selling days.
9. Creates a cleaning log showing how many rows were retained or removed at each step.

After cleaning, the notebook calculates:

- Daily quantity and revenue per item.
- Average units sold for each weekday/hour combination.

Revenue is calculated as quantity multiplied by price when a price column is available.

## 7. Exploratory Data Analysis

The notebook visualizes:

- Total units sold per day and a rolling seven-day average.
- The ten most popular menu items by units sold.
- Average hourly sales patterns by weekday.

These analyses help identify demand trends, menu popularity, and potential staffing or preparation peaks.

## 8. Feature Engineering

The forecasting table is built at a daily, per-item level.

Missing item/date combinations within the observed historical period are filled with zero sales. An additional future date is included for forecasting, and its unknown quantity remains missing.

### Historical lag features

The model uses sales from previous days:

- `lag_1`: sales one day earlier.
- `lag_2`: sales two days earlier.
- `lag_7`: sales seven days earlier.
- `lag_14`: sales fourteen days earlier.
- `lag_21`: sales twenty-one days earlier.
- `lag_28`: sales twenty-eight days earlier.

### Rolling statistics

- `roll_mean_7`: previous seven-day average.
- `roll_mean_14`: previous fourteen-day average.
- `roll_mean_28`: previous twenty-eight-day average.
- `roll_std_7`: previous seven-day sales standard deviation.
- `same_dow_avg_4w`: average of the previous four corresponding weekdays.

The rolling features use `shift(1)` so the current day's target is not included in its own features.

### Calendar and item features

- `dow`: day of the week.
- `is_weekend`: indicator for Friday, Saturday, or Sunday in this simulation.
- `month`: calendar month.
- `day_of_month`: day of the month.
- `item_id`: categorical numeric code for the menu item.

The complete feature list is stored in `FEATURES`.

## 9. Machine Learning Model

The notebook uses:

```python
HistGradientBoostingRegressor(
    loss="poisson",
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=24,
    min_samples_leaf=30,
    l2_regularization=1.0,
    categorical_features=[FEATURES.index("item_id")],
    early_stopping=False,
    random_state=0
)
```

### Why HistGradientBoostingRegressor?

It is a tree-based gradient boosting regression algorithm. It learns nonlinear relationships between historical demand, calendar effects, and menu items.

The Poisson loss is selected because the target represents non-negative sales counts.

The model is trained on historical daily item-level quantities. It predicts demand separately for each menu item and date.

## 10. Time-Based Training and Evaluation

The notebook avoids randomly shuffling time-series observations.

It creates chronological partitions:

- **Training set:** earlier observations used to fit the model.
- **Validation set:** the 28 days immediately before the test period; used to estimate typical forecast errors.
- **Test set:** the final 28 days; used for final backtesting.

This is more appropriate for forecasting than a random train/test split because future observations should not be used to predict the past.

### Evaluation metrics

The notebook calculates:

**Mean Absolute Error (MAE)**

The average absolute difference between actual and predicted demand.

**Weighted Absolute Percentage Error (WAPE)**

The total absolute forecast error divided by total actual demand, expressed as a percentage.

**Bias**

The average predicted quantity minus the actual quantity.

- Positive bias: the model tends to over-forecast.
- Negative bias: the model tends to under-forecast.

### Baseline comparisons

The machine learning model is compared against:

1. Same day last week (`lag_7`).
2. Seven-day rolling average (`roll_mean_7`).
3. Average of the previous four corresponding weekdays (`same_dow_avg_4w`).

The notebook also calculates per-item performance and plots actual versus predicted demand for the three highest-volume items.

## 11. Turning Forecasts into a Preparation Plan

A forecast represents expected demand, but a restaurant may want to prepare extra food to reduce stockouts.

The notebook uses:

```text
Preparation quantity = Forecast + z × Forecast error standard deviation
```

Where:

- `Forecast` is the model's predicted demand.
- `z` is the standard normal quantile associated with the chosen service level.
- `Forecast error standard deviation` is estimated separately for each item using validation residuals.

The result is rounded up and constrained to be non-negative.

### Service levels

The notebook evaluates service levels of:

- 50%
- 70%
- 80%
- 85%
- 90%
- 95%

A higher service level generally produces a larger preparation buffer. This can reduce shortages but increase leftover food.

### Business metrics

The simulation measures:

- **Waste:** units prepared but not sold.
- **Shortage:** units of demand that could not be fulfilled.

A scatter plot compares the waste-versus-shortage trade-off for the machine learning policies and baseline policies.

## 12. Final Forecast

After backtesting, the notebook refits the model using all available historical training rows and predicts demand for the next date included in the forecasting panel.

The forecast output includes:

- Date.
- Menu item.
- Predicted demand.
- Average sales for the last four corresponding weekdays.
- Sales on the same weekday last week.
- Estimated residual standard deviation.

It also calculates an 85% service-level preparation suggestion.

## 13. Model Explainability

The notebook uses Scikit-learn's permutation importance.

The method shuffles one feature at a time and measures how much the model's negative mean absolute error decreases in performance.

A larger increase in error after shuffling indicates that the feature was more useful to the model on the sampled test observations.

This provides a practical way to discuss which historical and calendar signals contribute to forecasting performance.

## 14. Exported Files

The notebook creates an `outputs/` directory containing:

| File | Description |
|---|---|
| `daily_sales.csv` | Clean daily sales aggregated by date and item |
| `hourly_sales.csv` | Average sales by weekday and hour |
| `forecast_tomorrow.csv` | Forecast and residual spread for the next date |
| `backtest.csv` | Actual demand, model forecast, baseline forecast, and error spread for the test period |
| `metrics.csv` | Overall evaluation metrics for the model and baselines |
| `cleaning_log.csv` | Summary of rows removed or retained during cleaning |
| `model.joblib` | Serialized final gradient boosting model |

## 15. Streamlit Dashboard

The notebook writes a Streamlit application to `app.py`.

The dashboard contains four tabs:

### 1. Sales Overview

- Total units sold.
- Average units per day.
- Revenue, when available, or number of menu items.
- Busiest weekday.
- Daily sales trend and seven-day average.
- Top menu items by units and revenue, or menu mix when revenue is unavailable.
- Data-cleaning summary.

### 2. Peak Hours & Patterns

- Heatmap of average sales by weekday and hour.
- Five busiest weekday/hour slots.
- Heatmap showing which menu items perform relatively better on particular weekdays.

### 3. Forecast Accuracy

- Machine learning WAPE.
- Baseline WAPE.
- Relative improvement over the same-day-last-week baseline.
- Evaluation metrics table.
- Actual versus predicted sales for a selected menu item.

### 4. Prep Plan for Tomorrow

- Adjustable service-level slider.
- Expected demand for each item.
- Likely demand range.
- Recommended preparation quantity.
- Last week's corresponding sales.
- Downloadable preparation sheet in CSV format.
- Historical waste and shortage comparisons.

The notebook launches Streamlit on port 8501 and uses Cloudflare Tunnel to expose the application from Google Colab.

## 16. How to Run the Project

### Option A: Google Colab

1. Upload `Restaurant_demand_forecasting_synthetic.ipynb` to Google Colab.
2. Open the notebook.
3. Run the cells from top to bottom.
4. Keep `USE_SYNTHETIC = True` to use the generated dataset, or configure your CSV and set it to `False`.
5. Wait for data cleaning, analysis, training, evaluation, and output export to finish.
6. Run the Streamlit dashboard cell.
7. Open the generated public tunnel link.

### Option B: Local Python Environment

Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib streamlit plotly
```

Run the notebook in Jupyter or VS Code.

After the notebook has generated the `outputs/` directory and `app.py`, launch the dashboard:

```bash
streamlit run app.py
```

The dashboard expects the exported CSV files to be present in the `outputs/` directory.

## 17. Important Implementation Notes and Limitations

- **Synthetic data:** Results on the default dataset demonstrate the workflow, not real-world restaurant performance. Real transaction data is needed to establish practical accuracy.
- **Forecast date wording:** The notebook generates historical data through yesterday and adds the next calendar date to the panel. Thus, the forecast is for the next date after the last observed sales date. The printed word "tomorrow" may be inaccurate depending on when the notebook is run.
- **Custom data mapping:** The uploaded CSV must contain suitable date/time, item, and quantity columns. The optional price field should contain usable numeric prices.
- **Returns:** Negative quantities are removed rather than modeled as separate return events.
- **Outliers:** The percentile-based cleaning rule may remove genuine bulk orders as well as data-entry errors. Review the cleaning log and threshold when using real data.
- **Zero-sales assumption:** Missing date/item combinations within the observed period are treated as zero sales. This is only appropriate when missing records genuinely mean no sales.
- **Service-level interpretation:** The normal-residual safety buffer is an approximation. Actual demand errors may not be normally distributed, independent, or equally variable over time. A nominal service level is not a guarantee of that percentage of fulfilled demand.
- **Feature importance:** Permutation importance describes predictive usefulness on the sampled test data; it does not establish causality.
- **Dashboard deployment:** The Cloudflare Tunnel cell is intended for a Colab demonstration. A production deployment should use an appropriate hosting and security configuration.
- **Model persistence:** The notebook exports the fitted model, but the dashboard currently reads precomputed CSV outputs rather than loading the saved model to make live predictions.

## 18. Future Improvements

Potential extensions include:

- Evaluate on real restaurant transaction datasets.
- Add weather, holidays, local events, and promotion information.
- Compare gradient boosting with Random Forest, XGBoost, LightGBM, and time-series models.
- Use rolling-origin backtesting across multiple forecast periods.
- Add prediction intervals or quantile forecasting.
- Include ingredient-level inventory requirements and supplier lead times.
- Add food costs and menu margins to estimate the financial impact of waste.
- Retrain the model automatically as new sales data arrives.
- Add multi-day forecasts and automated inventory alerts.

## 19. Conclusion

This project demonstrates an end-to-end restaurant demand forecasting workflow, from synthetic data generation and cleaning to machine learning, evaluation, preparation planning, and interactive visualization.

Its main business objective is to support better food preparation decisions by balancing expected demand, food waste, and the risk of running out of menu items.

---

**Project:** Restaurant Sales Analytics & Demand Forecasting  
**Notebook:** `Restaurant_demand_forecasting_synthetic.ipynb`  
**Primary model:** `HistGradientBoostingRegressor`  
**Dashboard:** Streamlit
