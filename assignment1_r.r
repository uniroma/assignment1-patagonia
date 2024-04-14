library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)
# Load the dataset
df <- read_csv('~/Downloads/current.csv')

# Extract transformation codes
transformation_codes <- data.frame(Series = names(df)[-1], Transformation_Code = as.numeric(df[1, -1]))

# Function to apply transformations based on the transformation code

mdiff <- function(x) {
    x - dplyr::lag(x, 1, default = NA)
}


apply_transformation <- function(series, code) {
  if (code == 1) {
    return(series)
  } else if (code == 2) {
    return(mdiff(series))
  } else if (code == 3) {
    return(mdiff(mdiff(series)))
  } else if (code == 4) {
    return(log(series))
  } else if (code == 5) {
    return(mdiff(log(series)))
  } else if (code == 6) {
    return(mdiff(mdiff(log(series))))
  } else if (code == 7) {
    return(mdiff(series) / dplyr::lag(series, 1) - 1)
  } else {
    stop("Invalid transformation code")
  }
}

# Applying the transformations to each column in df_cleaned based on transformation_codes
for (i in 1:nrow(transformation_codes)) {
  series_name <- transformation_codes$Series[i]
  code <- transformation_codes$Transformation_Code[i]
  df[[series_name]] <- apply_transformation(as.numeric(df[[series_name]]), code)
}

df_cleaned <- df[-c(1:3), ]

# Plot transformed series
series_to_plot <- c('INDPRO', 'CPIAUCSL', 'TB3MS')
series_names <- c('Industrial Production', 'Inflation (CPI)', '3-month Treasury Bill rate')

 plot_data <- df_cleaned %>%
   select(sasdate, all_of(series_to_plot)) %>%
   pivot_longer(-sasdate, names_to = "series", values_to = "value") %>%
   mutate(sasdate = mdy(sasdate),
          series_name = factor(series, levels = series_to_plot, labels = series_names))

 ggplot(plot_data, aes(x = sasdate, y = value, color = series_name)) +
   geom_line() +
   facet_wrap(~series_name, scales = "free", ncol=1) +
   theme_minimal() +
   labs(x = "Year", y = "Transformed Value") +
   theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")





# Prepare data for estimation
Yraw <- df_cleaned$INDPRO
Xraw <- df_cleaned %>% select(CPIAUCSL, TB3MS)

num_lags <- 4  # this is p
num_leads <- 1  # this is h

X <- data.frame(Ones = rep(1, nrow(df_cleaned)))

for (lag in 0:num_lags) {
  X[paste0('INDPRO_lag', lag)] <- dplyr::lag(Yraw, lag)
}

for (col in names(Xraw)) {
  for (lag in 0:num_lags) {
    X[paste0(col, '_lag', lag)] <- dplyr::lag(Xraw[[col]], lag)
  }
}

y <- dplyr::lead(Yraw, num_leads)

## Getting the last row fro forecasting
X_T <- as.matrix(tail(X, 1))

# Removing NA rows (due to lagging/leading)
complete_cases <- complete.cases(X, y)
X <- X[complete_cases, ]
y <- y[complete_cases]

# Estimation and forecast

y <- as.vector(y)
X <- as.matrix(X)

beta_ols <- solve(crossprod(X), crossprod(X, y))

# Produce the One step ahead forecast
forecast <- (X_T %*% beta_ols) * 100
