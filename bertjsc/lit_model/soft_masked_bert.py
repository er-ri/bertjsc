import torch
import torch.nn as nn

from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from typing import Optional
from .train_config import TrainConfig

class LitSoftMaskedBert(TrainConfig):

    def __init__(self, card, mask_token_id, vocab_size):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlbert = BertForMaskedLM.from_pretrained(card)

        # Word embedding
        self.embeddings = self.mlbert.bert.embeddings

        # Detection neural network
        self.bidirectional_gru = nn.GRU(input_size=self.mlbert.config.hidden_size, 
                                        hidden_size=self.mlbert.config.hidden_size, num_layers=1,
                                        bidirectional=True, batch_first=True)

        self.linear = nn.Linear(self.mlbert.config.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # Correction neural network
        self.encoder = self.mlbert.bert.encoder
        self.cls = self.mlbert.cls
        
        # Loss function
        self.det_criterion = nn.BCELoss()
        self.cor_criterion = nn.CrossEntropyLoss()
        
        # Coefficient
        self.coef = 0.8
        
    def forward(self, 
            input_ids: torch.Tensor, 
            token_type_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            output_ids: Optional[torch.Tensor] = None, 
            det_labels: Optional[torch.Tensor] = None
        ):
        embeddings = self.embeddings(input_ids=input_ids,
                                    token_type_ids=token_type_ids)
        
        # Detection
        gru_output, _ = self.bidirectional_gru(embeddings)
        prob = self.sigmoid((self.linear(gru_output)))
        
        masked_e = self.embeddings(torch.tensor([[self.mask_token_id]], dtype=torch.long).to(self.device))
        soft_masked_embeddings = prob * masked_e + (1 - prob) * embeddings
        
        # About `extended_attention_mask`
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L852
        extended_attention_mask: torch.Tensor = self.mlbert.get_extended_attention_mask(attention_mask,
                                                                                input_ids.size())
        
        # Correction
        bert_out = self.encoder(hidden_states=soft_masked_embeddings,
                                attention_mask=extended_attention_mask)
        h = bert_out[0] + embeddings
        
        prediction_scores = self.cls(h)
        
        loss = None
        if output_ids is not None and det_labels is not None:
            det_loss = self.det_criterion(prob.squeeze(), det_labels)
            cor_loss = self.cor_criterion(prediction_scores.view(-1, self.vocab_size), output_ids.view(-1))
            loss = self.coef * cor_loss + (1 - self.coef) * det_loss
        
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores
        )
        
