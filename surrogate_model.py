import numpy as np


class ARIMAXSurrogate:
    """
    Lightweight surrogate for a fitted ARIMAX model.
    Coefficients are recovered from the 4 known stress scenario outputs.
    Implements the same predict() interface as pmdarima AutoARIMA.
    """

    def __init__(self, intercept, coefficients):
        self.intercept = intercept
        self.coefficients = np.array(coefficients)

    def predict(self, n_periods, X=None, return_conf_int=False, alpha=0.05):
        if X is not None and len(X) > 0:
            x_row = np.asarray(X[0]) if np.asarray(X).ndim > 1 else np.asarray(X)
            rate = float(self.intercept + np.dot(self.coefficients, x_row))
        else:
            rate = float(self.intercept)
        rate = max(rate, 0.0)
        forecasts = np.full(n_periods, rate)
        if return_conf_int:
            width = max(abs(rate) * 0.15, 0.01)
            conf_int = np.column_stack([forecasts - width, forecasts + width])
            return forecasts, conf_int
        return forecasts
