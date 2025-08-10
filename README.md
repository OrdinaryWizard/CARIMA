# CARIMA (C ARIMA)
---
An implementation of an ARIMA model written completely in C. It uses gradient descent to tune the parameters. It also includes a normalisation function if you need it.
## Usage
---
It's pretty straight forward. Include the file arima.c into your C file and kabam u got it.

```ARIMA(X, n, p, d, q, epochs, learning_rate, verbose)```

| Argument        |                                                                          |
| --------------- | ------------------------------------------------------------------------ |
| `X`             | the time series you want to pass in (double array)                       |
| `n`             | the size of the time series (length of the array)                        |
| `p, d, q`       | the standard ARIMA parameters                                            |
| `epochs`        | how many training steps the model should train for                       |
| `learning_rate` | how much to tune the parameters each epoch                               |
| `verbose`       | integer boolean (1 or 0), whether to display the training process or not |
Additionally you can also use the `ARMA` function, without the excluding the `d` parameter of course.

### Example Usage

```
int main() {
    // Example time series data (e.g., stock prices or any series)
    double data[] = {
        112, 118, 132, 129, 121, 135, 148, 148,
        136, 119, 104, 118, 115, 126, 141, 135,
        125, 149, 170, 170, 158, 133, 114, 140
    };
    int n = sizeof(data) / sizeof(data[0]);

    // Optional: Normalize the data to zero mean and unit variance
    double mean, std;
    normalise(data, n, &mean, &std);

    // ARIMA hyperparameters
    int p = 2;             // AR order
    int d = 1;             // Differencing order
    int q = 2;             // MA order
    int epochs = 10;      // Number of training iterations
    double learning_rate = 0.001;
    int verbose = 1;

    // Call ARIMA function to get forecast (returns the next predicted value)
    double forecast = ARIMA(data, n, p, d, q, epochs, learning_rate, verbose);

    printf("\nForecasted next value (normalized scale): %lf\n", forecast);

    // If you normalized data, you might want to convert forecast back to original scale
    double forecast_original = forecast * std + mean;
    printf("Forecasted next value (original scale): %lf\n", forecast_original);

    return 0;
}
```

**Output**:

```
Epoch: 2, Loss: -0.174601
Epoch: 3, Loss: -0.465578
Epoch: 4, Loss: 0.814737
Epoch: 5, Loss: 0.756868
Epoch: 6, Loss: 0.000790
Epoch: 7, Loss: -0.697323
Epoch: 8, Loss: -0.989059
Epoch: 9, Loss: -0.872883
Epoch: 10, Loss: 0.816697
Epoch: 11, Loss: -0.178893

Forecasted next value (normalized scale): 0.400686
Forecasted next value (original scale): 140.051268
```

## Intention
---
I wanted to use ARIMA for a personal project in Python, but I found the `statsmodel` ARIMA implementation was too slow when conducting large backtests. So I quickly wrote this implementation in C using gradient descent to tune parameters, and it performs much faster. Of course `statsmodels` probably has better parameter tuning, but in practice I found both implementations to be roughly equal in accuracy.
