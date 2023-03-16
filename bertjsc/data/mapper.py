"""Pytorch map-style datasets for MLM(Masked-Language Modelling) BERT and Soft-Masked BERT.
See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#getitem
"""

import torch

class MaskedDataset(torch.utils.data.Dataset):
    r"""
    Description:
        Dataset mapper for MLM(Masked-Language Modelling) BERT.
    """

    def __init__(self, tokenizer, dataset, max_length=32):
        """
        Parameters:
            tokenizer: Japanese tokenizer
            dataset: pd.DataFrame
        Note:
            Check Dataset contents: `next(iter(dataset))`
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        
    def __getitem__(self, idx):

        encodings = self.tokenizer.encode_plus(self.dataset.iloc[idx]['pre_text'], 
            truncation=True, padding='max_length', max_length=self.max_length)
        encodings['labels'] = self.tokenizer.encode_plus(self.dataset.iloc[idx]['post_text'], 
            truncation=True, padding='max_length', max_length=self.max_length)['input_ids']
        encodings = { k: torch.tensor(v) for k, v in encodings.items() }

        return encodings

    def __len__(self):
        return len(self.dataset)


class SoftMaskedDataset(torch.utils.data.Dataset):
    r"""
    Description:
        Dataset mapper for Soft-Masked-Bert.
    """

    def __init__(self, tokenizer, dataset, max_length=32):
        """
        Parameters:
            tokenizer: Japanese tokenizer
            dataset: pd.DataFrame
        Note:
            Check Dataset contents: `next(iter(dataset))`
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        
    def __getitem__(self, idx):
        
        sample = dict()
        sample = self.tokenizer.encode_plus(self.dataset.iloc[idx]['pre_text'], 
                    truncation=True, padding='max_length', max_length=self.max_length)
        sample['output_ids'] = self.tokenizer.encode_plus(self.dataset.iloc[idx]['post_text'], 
                    truncation=True, padding='max_length', max_length=self.max_length)['input_ids']
        
        # Construct detection labels for every token, where '1' represents correct and '0' represents wrong, respectively. 
        sample['det_labels'] = [ float(1) if i == j else float(0) for i, j in zip(sample['input_ids'], sample['output_ids'])]
        sample = { k: torch.tensor(v) for k, v in sample.items() }
        
        return sample

    def __len__(self):
        return len(self.dataset)