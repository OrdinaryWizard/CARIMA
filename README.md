# CARIMA (C ARIMA)
An implementation of an ARIMA model written completely in C. It uses gradient descent to tune the parameters.

## Usage
---
It's pretty straight forward. Include the file arima.c into your C file and kabam u got it.
ARIMA(X, n, p, d, q, epochs, learning_rate, verbose)

X = the time series you want to pass in (double array)
n = the size of the time series (length of the array)
p, d, q = the standard ARIMA parameters
epochs = how many training steps the model should train for
learning_rate = how much to tune the parameters each epoch
verbose = integer boolean (1 or 0), whether to display the training process or not

Additionally you can also use the ARMA function, without the d parameter of course

## Intention
---
I wanted to use ARIMA for a personal project in Python, but I found the statsmodel was too slow when conducting large backtests. So I quickly wrote this implementation in C using gradient descent to tune parameters, and it performs much faster. 
