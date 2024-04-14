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

############################################################################################################
## Plot transformed series
############################################################################################################
import matplotlib.pyplot as plt         
import matplotlib.dates as mdates       

series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']         
series_names = ['Industrial Production',                 
                'Inflation (CPI)',                        
                '3-month Treasury Bill rate']            


# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))       

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:                                
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y') 
        ax.plot(dates, df_cleaned[series_name], label=plot_title)        
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))           
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))         
        ax.set_title(plot_title)                                         
        ax.set_xlabel('Year')                                            
        ax.set_ylabel('Transformed Value')                               
        ax.legend(loc='upper left')                                      
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right') 
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout() 
plt.show()         

############################################################################################################
## Create y and X for estimation of parameters
############################################################################################################

Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

## Number of lags and leads
num_lags  = 4  ## this is p
num_leads = 1  ## this is h

X = pd.DataFrame()
## Add the lagged values of Y
col = 'INDPRO'
for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Yraw.shift(lag)
## Add the lagged values of X
for col in Xraw.columns:
    for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
## Add a column on ones (for the intercept)
X.insert(0, 'Ones', np.ones(len(X)))

## X is now a DataFrame with lagged values of Y and X
X.head()

## Y is now the leaded target variable
y = Yraw.shift(-num_leads)


############################################################################################################
## Estimation and forecast
############################################################################################################

## Save last row of X (converted to numpy)
X_T = X.iloc[-1:].values

## Subset getting only rows of X and y from p+1 to h-1
## and convert to numpy array
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

## Import the solve function from numpy.linalg
from numpy.linalg import solve

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X.T @ X, X.T @ y)

## Produce the One step ahead forecast
## % change month-to-month of INDPRO
forecast = X_T@beta_ols*100
