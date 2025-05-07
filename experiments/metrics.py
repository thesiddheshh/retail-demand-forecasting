from sklearn.metrics import mean_squared_error

def compute_rmse(actual, predicted):
    """
    Compute Root Mean Squared Error.
    """
    rmse = mean_squared_error(actual, predicted, squared=False)
    return rmse
