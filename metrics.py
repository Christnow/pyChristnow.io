from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def report(y_pred, y_true):
    a = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    output_dict = False
    score = classification_report(y_true=y_true,
                                  y_pred=y_pred,
                                  target_names=target_names,
                                  output_dict=output_dict)
    return a, p, r, f1, score
