#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Optional: normalize data to zero mean and unit variance
void normalise(double *x, int n, double *mean, double *std) {
    *mean = 0;
    *std = 0;
    for (int i = 0; i < n; i++) *mean += x[i];
    *mean /= n;

    for (int i = 0; i < n; i++) *std += (x[i] - *mean) * (x[i] - *mean);
    *std = sqrt(*std / n);

    for (int i = 0; i < n; i++) x[i] = (x[i] - *mean) / *std;
}

double ARMA(double X[], int n, int p, int q, int epochs, double learning_rate, int verbose) {
    if (epochs + p > n || n <= p || n <= q) {
        fprintf(stderr, "Invalid data shape for ARMA\n");
        return NAN;
    }

    double *alpha = calloc(p, sizeof(double));
    double *epsilon = calloc(n, sizeof(double));
    double *theta = calloc(q, sizeof(double));

    for (int epoch = p; epoch < epochs + p; epoch++) {
        double alphas = 0;
        double epsilons = epsilon[epoch];

        for (int i = 1; i <= p; i++) alphas += alpha[i - 1] * X[epoch - i];
        for (int i = 1; i <= q; i++) epsilons += theta[i - 1] * epsilon[epoch - i];

        double forecast = alphas + epsilons;
        double loss = X[epoch] - forecast;
        epsilon[epoch] = loss;

        if (isnan(loss) || isinf(loss)) {
            fprintf(stderr, "Numerical instability at epoch %d, loss: %lf\n", epoch, loss);
            break;
        }

        // Optional: gradient clipping
        const double CLIP = 10.0;

        for (int i = 1; i <= p; i++) {
            double grad = loss * X[epoch - i];
            if (grad > CLIP) grad = CLIP;
            if (grad < -CLIP) grad = -CLIP;
            alpha[i - 1] += grad * learning_rate;
        }
        for (int i = 1; i <= q; i++) {
            double grad = loss * epsilon[epoch - i];
            if (grad > CLIP) grad = CLIP;
            if (grad < -CLIP) grad = -CLIP;
            theta[i - 1] += grad * learning_rate;
        }

        if (verbose) printf("Epoch: %d, Loss: %lf\n", epoch, loss);
    }

    double alphas_sum = 0;
    double epsilon_sum = 0;
    for (int i = 1; i <= p; i++) alphas_sum += alpha[i - 1] * X[n - i];
    for (int i = 1; i <= q; i++) epsilon_sum += theta[i - 1] * epsilon[n - i];

    free(alpha);
    free(epsilon);
    free(theta);

    return alphas_sum + epsilon_sum;
}

double ARIMA(double X[], int n, int p, int d, int q, int epochs, double learning_rate, int verbose) {
    // Allocate memory for differencing
    int total_size = 0;
    for (int i = 0; i <= d; i++) total_size += (n - i);
    double *diff_store = malloc(total_size * sizeof(double));
    double **D = malloc((d + 1) * sizeof(double*));

    D[0] = diff_store;
    memcpy(D[0], X, n * sizeof(double));

    int offset = n;
    for (int i = 1; i <= d; i++) {
        D[i] = diff_store + offset;
        int len_prev = n - (i - 1);
        for (int j = 0; j < len_prev - 1; j++) {
            D[i][j] = D[i - 1][j + 1] - D[i - 1][j];
        }
        offset += len_prev - 1;
    }

    int final_len = n - d;
    double restore = ARMA(D[d], final_len, p, q, epochs, learning_rate, verbose);

    // Inverse differencing
    for (int i = d - 1; i >= 0; i--) {
        restore += D[i][n - d + (d - 1 - i)];
    }

    free(diff_store);
    free(D);
    return restore;
}
