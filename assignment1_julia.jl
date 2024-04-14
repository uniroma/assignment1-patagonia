using CSV
using DataFrames
using Dates
using LinearAlgebra
using Plots

# Load the dataset
df = CSV.read("/Users/gragusa/Downloads/current.csv", DataFrame)

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df[2:end, :]
date_format = "mm/dd/yyyy"
df_cleaned[!, :sasdate] = Dates.Date.(df_cleaned[!, :sasdate], date_format)


df_original = copy(df_cleaned)
df_cleaned = coalesce.(df_cleaned, NaN)
df_original = coalesce.(df_original, NaN)
## Create a DataFrame with the transformation codes
transformation_codes = DataFrame(Series = names(df)[2:end], 
                                 Transformation_Code = collect(df[1, 2:end]))

# Function to apply transformations based on the transformation code
function apply_transformation(series, code)
    if code == 1
        # No transformation
        return series
    elseif code == 2
        # First difference
        return mdiff(series)
    elseif code == 3
        # Second difference
        return mdiff(mdiff(series))
    elseif code == 4
        # Log
        return log.(series)
    elseif code == 5
        # First difference of log
        return mdiff(log.(series))
    elseif code == 6
        # Second difference of log
        return mdiff(mdiff(log.(series)))
    elseif code == 7
        # Delta (x_t/x_{t-1} - 1)
        return series ./ lag(series, 1) .- 1
    else
        throw(ArgumentError("Invalid transformation code"))
    end
end


# Helper function to lag a series; 
# Julia does not have a built-in lag function like R or pandas
function lag(v::Vector, l::Integer)
    nan = [NaN for _ in 1:l]
    return [nan; v[1:(end-l)]]
end

function lead(v::Vector, l::Integer)
    nan = [NaN for _ in 1:l]
    return [v[(l+1):end]; nan]
end

## mdiff function to calculate the first difference of a series
## keeping the missing values
function mdiff(v::Vector)
    return v .- lag(v, 1)
end


# Applying the transformations to each column in df_cleaned based on transformation_codes
for row in eachrow(transformation_codes)
    series_name = Symbol(row[:Series])
    code = row[:Transformation_Code]
    @show series_name, code
    df_cleaned[!, series_name] = apply_transformation(df_cleaned[!, series_name], code)
end


## The transformation create missing values at the top
## These remove the missing values at the top of the dataframe
df_cleaned = df_cleaned[3:end, :]

# Plotting with customized x-axis ticks
p1 = plot(df_cleaned.sasdate, df_cleaned.INDPRO, label="Industrial Production", legend=:none, xlabel="Date", ylabel="INDPRO", title="Industrial Production")
p2 = plot(df_cleaned.sasdate, df_cleaned.CPIAUCSL, label="CPI", legend=:none, xlabel="Date", ylabel="CPIAUCSL", title="Consumer Price Index")

# Combine the plots into a 2x1 layout
plot(p1, p2, layout=(2, 1), size=(800, 600))


## To save:
## Plots.savefig(p, "output.png")

## Obtain the forecast


# Extract the series and convert them into Float64 if they are not already
Y = df_cleaned[!, :INDPRO]
X = Matrix(df_cleaned[!, [:CPIAUCSL, :TB3MS]])


num_leads = 1 # One-step ahead
num_lags = 4


# Create lagged versions of Y and X, and handle the dropping of missing values accordingly
Y_target = lead(Y, 1)
Y_lagged = hcat([lag(Y, i) for i in 0:num_lags]...)
X_lagged = hcat([lag(X[:, j], i) for j in axes(X, 2) for i in 0:num_lags]...)

## Add column of ones for the constant term
X = hcat(ones(size(Y_lagged, 1)), Y_lagged, X_lagged)

## For the forecast last row of the X which will get removed later
X_T = X[end, :]

y  = Y_target[num_lags+1:(end-num_leads)]
X_ = X[num_lags+1:(end-num_leads), :]

# OLS estimator using the Normal Equation
beta_ols = X_ \ y

# Produce the One-step ahead forecast and convert it to percentage
forecast = (X_T' * beta_ols) * 100



