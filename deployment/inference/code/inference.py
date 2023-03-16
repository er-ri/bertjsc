import os
import torch

from transformers import BertJapaneseTokenizer

from bertjsc import predict_with_json_result
from bertjsc.lit_model import LitBertForMaskedLM

MODEL_CARD = "cl-tohoku/bert-base-japanese-whole-word-masking"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_CARD)

def model_fn(model_dir):
    """
    Load the model for inference
    """
    model = LitBertForMaskedLM(MODEL_CARD)
    with open(os.path.join(model_dir, 'lit-bert-for-maskedlm-230126.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model.to(device)

def input_fn(request_body, request_content_type):
    """
    Description:
        Deserialize and prepare the prediction input
    Args:
        request_body: byte buffer
        request_content_type: str
    Note:
    * A default input_fn that can handle JSON, CSV and NPZ formats.
    """
    input_object = request_body
    return input_object

def predict_fn(input_object, model):
    """
    Apply model to the incoming request
    """
    prediction = predict_with_json_result(model, tokenizer, device, input_object)

    return prediction

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    return prediction