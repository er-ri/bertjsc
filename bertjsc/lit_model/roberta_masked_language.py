import torch

from transformers import RobertaForMaskedLM

from typing import Optional
from .train_config import TrainConfig

class LitRobertaForMaskedLM(TrainConfig):
    """
    RoBERTa Japanese model
    Ref: https://huggingface.co/rinna/japanese-roberta-base
    """
    
    def __init__(self, card):
        super().__init__()
        self.automatic_optimization = True
        self.mlroberta = RobertaForMaskedLM.from_pretrained(card)
        
    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
        ):

        output = self.mlroberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        return output