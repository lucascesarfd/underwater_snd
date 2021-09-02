
def get_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def get_precision(tp, fp):
    return tp / (tp + fp)

def get_recall(tp, fn):
    return tp / (tp + fn)

def get_f1(tp, tn, fp, fn):
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    return 2 * (precision * recall) / (precision + recall)

