from sklearn.metrics import mean_squared_error

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, squared=False)

    # pearson
    # shear
    # bin_acc
    return {"mse": mse}