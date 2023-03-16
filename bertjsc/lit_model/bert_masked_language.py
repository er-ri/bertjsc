import torch

from transformers import BertForMaskedLM

from typing import Optional
from .train_config import TrainConfig

class LitBertForMaskedLM(TrainConfig):
        
    def __init__(self, card):
        super().__init__()
        self.automatic_optimization = True
        self.mlbert = BertForMaskedLM.from_pretrained(card)
        
    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
        ):
        # BerForMaskedLM Loss(When `labels` is not None): CrossEntropyLoss()
        # See: https://github.com/huggingface/transformers/blob/31d452c68b34c2567b62924ee0df40a83cbc52d5/src/transformers/models/bert/modeling_bert.py#L1302
        output = self.mlbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        return output
    