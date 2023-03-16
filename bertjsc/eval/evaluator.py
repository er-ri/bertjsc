import torch
import numpy as np
from tqdm import tqdm

def show_model_performance(model, tokenizer, df):
    r"""
    Description:
        Show model performance with the given test dataset.
    Args:
        model: Masked-Language-Model
        tokenizer: Japanese Tokenizer
        df: pd.DataFrame
    Return:
        Metrics(dict)
    """
    # Calculate model's detection and correction performance
    # TP(True Positive): Contains errors, and the prediction is correct.
    # FP(False Positive): Contains errors, but the prediction is incorrect or not performed.
    # TN(True Negative): No errors, and the prediction is not performed.
    # FN(False Negative): No errors, but the prediction is performed.
    detect_tp = 0
    detect_fp = 0
    correct_tp = 0
    correct_fp = 0
    tn = 0
    fn = 0
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        pre_text = row['pre_text']
        post_text = row['post_text']
        
        detect_row_tp, detect_row_fp, correct_row_tp, correct_row_fp = calculate_positive(pre_text, post_text, model, tokenizer)
        row_tn, row_fn = calculate_negative(post_text, model, tokenizer)
        
        detect_tp += detect_row_tp
        detect_fp += detect_row_fp
        correct_tp += correct_row_tp
        correct_fp += correct_row_fp
        tn = tn + row_tn
        fn = fn + row_fn
    
    detect_metrics = calculate_metrics(detect_tp, detect_fp, tn, fn)
    correct_metrics = calculate_metrics(correct_tp, correct_fp, tn, fn)

    metrics = {
        "Detection": detect_metrics,
        "Correction": correct_metrics
    }

    return metrics

def calculate_positive(pre_text, post_text, model, tokenizer):
    r"""
    Description:
        Caculate TP/FP for the `pre_text`(contains errors).
    Args:
        `pre_test`: str
        `post_test`: str
        `model`: pytorch model
        `tokenizer`: Japanese tokenizer
    Return:
        The detection and correction TP/FP.
    """
    encodings = tokenizer(pre_text, max_length=128, truncation=True, return_tensors="pt")
    pre_ids_list = encodings['input_ids'].tolist()[0]
    
    predict_ids_list = get_predict_ids_list(model, encodings)
    
    post_ids = tokenizer(post_text, max_length=128, truncation=True, return_tensors="pt")['input_ids']
    post_ids_list = post_ids.tolist()[0]
    
    correct_tp=0
    correct_fp=0
    if post_ids_list[1:-1] == predict_ids_list[1:-1]:
        correct_tp = 1
    else:
        correct_fp = 1
        
    detect_tp=0
    detect_fp=0
    predict_ids_list = set_detect_position(pre_ids_list, predict_ids_list)
    post_ids_list = set_detect_position(pre_ids_list, post_ids_list)
    if post_ids_list[1:-1] == predict_ids_list[1:-1]:
        detect_tp = 1
    else:
        detect_fp = 1
    
    return detect_tp, detect_fp, correct_tp, correct_fp

def get_predict_ids_list(model, encodings):
    r"""
    Description:
        Perform model prediction for the `encodings`.
    Args:
        `model`: pytorch model
        `encodings`: Output of japanese tokenizer
    Return:
        The result of the id list for the `encodings`.
    """
    with torch.no_grad():
        output = model(**encodings)
    predict_ids = torch.topk(output.logits[0], 1, dim=1).indices.tolist()

    predict_ids_list = []
    for ids in predict_ids:
        predict_ids_list.append(ids[0])
        
    return predict_ids_list

def set_detect_position(error_ids_list, ids_list):
    r"""
    Description:
        Set element to 0 when the both lists have the same value, else set to 1.
    """
    # Set to 0 by `xor` operator.
    ids_list = np.bitwise_xor(error_ids_list, ids_list)
    # Set to 1 for the null 0 element.
    ids_list[ids_list!=0] = 1

    return ids_list.tolist()

def calculate_negative(post_text, model, tokenizer):
    r"""
    Description:
        Caculate TN/FN for the `post_text`(no errors).
    Args:
        `post_test`: str
        `model`: pytorch model
        `tokenizer`: Japanese tokenizer
    Return:
        The detection and correction TN/FN(Both have the same value).
    """
    encodings = tokenizer(post_text, max_length=128, truncation=True, return_tensors="pt")
    
    ids_list = encodings['input_ids'].tolist()[0]
    
    predict_ids_list = get_predict_ids_list(model, encodings)
        
    tn=0
    fn=0
    if ids_list[1:-1] == predict_ids_list[1:-1]:
        tn = 1
    else:
        fn = 1
        
    return tn, fn

def calculate_metrics(tp, fp, tn, fn):
    r"""
    Description:
        Caculate the `Accuracy`, `Precision`, `Recall` and `F1 Score`.
    Args:
        `tp`: int(True Positive)
        `fp`: int(False Positive)
        `tn`: int(True Negative)
        `fn`: int(False Negative)
    Return:
        Metrics(dict)
    """
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = 2*(prec*rec)/(prec+rec)
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }
    
    return metrics