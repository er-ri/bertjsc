r"""Training Script: Masked-Language Model for Japanese Spelling Error Correction task.
"""
import sys
[sys.path.append(i) for i in ['.', '..']]

import torch
from transformers import BertJapaneseTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from bertjsc.lit_model import LitBertForMaskedLM
from bertjsc.data import read_jwtd, get_ml_trainable4jwtd, MaskedDataset
from bertjsc.eval import show_model_performance

def main():
    # Initialization, replace with `LitSoftMaskedBert` when training Soft-masked BERT
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    model = LitBertForMaskedLM("cl-tohoku/bert-base-japanese-whole-word-masking")

    # Load training data
    df = read_jwtd("./dataset/jwtd_v1.0/train.jsonl")
    df_test = read_jwtd("./dataset/jwtd_v1.0/test.jsonl")

    # Create a smaller dataset
    df = df[:50]
    df_test = df[:10]

    # Extract the trainable data
    error_types = ['substitution', 'deletion', 'insertion', 'kanji-conversion']
    max_unmatch = 3

    df = get_ml_trainable4jwtd(df, tokenizer, error_types, max_unmatch)
    df_test = get_ml_trainable4jwtd(df_test, tokenizer, error_types, max_unmatch)

    # Split data to training & evaluation.
    df_train, df_val = train_test_split(df, test_size=0.2)

    print(f'Length of df_train: {len(df_train)}')
    print(f'Length of df_val: {len(df_val)}')
    print(f'Length of df_test: {len(df_test)}')

    # Convert to torch Dataset, `SoftMaskedDataset` for Soft-masked BERT.
    train_dataset = MaskedDataset(tokenizer, df_train)
    val_dataset = MaskedDataset(tokenizer, df_val)

    logger = TensorBoardLogger(save_dir="tb_logs", name='mlbert')
    # Train on CPU
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    trainer = Trainer(max_epochs=6, logger=logger, fast_dev_run=True, log_every_n_steps=50)

    # Train on GPU
    # dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=True)
    # dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=128,  num_workers=4, pin_memory=True, shuffle=False)
    # trainer = Trainer(accelerator='gpu', devices=1, 
    #                 max_epochs=6, 
    #                 enable_checkpointing=False, 
    #                 logger=logger, log_every_n_steps=50, fast_dev_run=False)

    model.train()
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # Comment out this line if you need to save the fine-tuned model
    # torch.save(model.state_dict(), '<path/to/save>/lit-bert-for-maskedlm-yymmdd.pth')

    # Caculate model's performance
    metrics = show_model_performance(model, tokenizer, df_test)
    print(metrics)

    print('Completed.')

if __name__ == '__main__':
    main()