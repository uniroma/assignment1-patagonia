import pandas as pd
from numpy.linalg import solve
import numpy as np

# Load the dataset
df = pd.read_csv('~/Downloads/current.csv')
print(df)

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

## df_cleaned contains the data cleaned
df_cleaned

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

## transformation_codes contains the transformation codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$



# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned.head()

## Plot the transformed series
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production', 'Inflation (CPI)', 'Federal Funds Rate']
# 'INDPRO'   for Industrial Production, 
# 'CPIAUCSL' for Inflation (Consumer Price Index), 
# 'TB3MS'    3-month treasury bill.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        # Formatting the x-axis to show only every five years
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        # Improve layout of date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.show()

Y = df_cleaned['INDPRO'].dropna()
X = df_cleaned[['CPIAUCSL', 'FEDFUNDS']].dropna()

h = 1 ## One-step ahead
p = 4
r = 4

Y_target = Y.shift(-h).dropna()
Y_lagged = pd.concat([Y.shift(i) for i in range(p+1)], axis=1).dropna()
X_lagged = pd.concat([X.shift(i) for i in range(r+1)], axis=1).dropna()
common_index = Y_lagged.index.intersection(Y_target.index)
common_index = common_index.intersection(X_lagged.index)

## This is the last row needed to create the forecast
X_T = np.concatenate([[1], Y_lagged.iloc[-1], X_lagged.iloc[-1]])

## Align the data
Y_target = Y_target.loc[common_index]
Y_lagged = Y_lagged.loc[common_index]
X_lagged = X_lagged.loc[common_index]

X_reg = pd.concat([X_lagged, Y_lagged], axis = 1)



X_reg = pd.concat([X_lagged, Y_lagged], axis=1)
X_reg_np = np.concatenate([np.ones((X_reg.shape[0], 1)), X_reg.values], axis=1)
Y_target_np = Y_target.values

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X_reg_np.T @ X_reg_np, X_reg_np.T @ Y_target_np)

## Produce the One step ahead forecast
## % change month-to-month INDPRO
print(X_T)
forecast = X_T@beta_ols*100
print(forecast)
print(beta_ols)
