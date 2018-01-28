import keras.backend as K

def fscore(pred, label):
    """
    Metric for calculating F1 score. An F1 score is a good way to measure when there is a class imbalance.
    It can be interpreted as a weighted average of precision and recall.

    Precision maxes out when there are few false positives.
    Recall maxes out when there are few false negatives.
    """
    false_negatives = K.sum(K.round(K.clip(label - pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip(pred - label, 0, 1)))
    true_positives = K.sum(K.round(pred * label))
    true_negatives = K.sum(K.round((1 - pred) * (1 - label)))

    pres = true_positives / (true_positives + false_positives + 1E-9)
    rec = true_positives / (true_positives + false_negatives + 1E-9)
    fscore = 2 * pres * rec / (pres + rec + 1E-9)
    return fscore


