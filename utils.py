import numpy as np

def compute_metrics(eval_pred, label2id):
    preds, labels = eval_pred

    import pdb; pdb.set_trace()

    colwise_rmse = np.sqrt(np.mean((labels - preds) ** 2, axis=0))
    mean_rmse = np.mean(colwise_rmse)

    metrics = {}

    for label, id_ in label2id.items():
        metrics[f"{label}_rmse"] = colwise_rmse[:, id_]

    metrics["mcrmse"] = mean_rmse

    return metrics
