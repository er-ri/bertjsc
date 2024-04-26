r"""Training Script: Masked-Language Model for Japanese Spelling Error Correction task.
"""
import sys
[sys.path.append(i) for i in ['.', '..']]

import torch
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from bertjsc.lit_model import LitRobertaForMaskedLM
from bertjsc.data import read_jwtd, get_ml_trainable4jwtd, MaskedDataset
from bertjsc.eval import show_model_performance


TRAIN_DATASET = "path/to/jwtd_v1.0/train.jsonl"
TEST_DATASET = "path/to/jwtd_v1.0/test.jsonl"
SAVE_MODEL_PATH = "path/to/fine_tuned_model.pth"
LOG_DIR = "tb_logs"     # TensorBoard log, use `tensorboard --logdir=tb_logs` to view


def main():
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base", use_fast=False)
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model = LitRobertaForMaskedLM("rinna/japanese-roberta-base")

    # Load training data
    df = read_jwtd(TRAIN_DATASET)
    df_test = read_jwtd(TEST_DATASET)

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

    logger = TensorBoardLogger(save_dir=LOG_DIR, name='roberta')

    # Training
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=128,  num_workers=4, pin_memory=True, shuffle=False)
    trainer = Trainer(accelerator='gpu', devices=1, 
                    max_epochs=6, 
                    enable_checkpointing=False, 
                    logger=logger, log_every_n_steps=50, fast_dev_run=False)

    model.train()
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # Save the fine-tuned model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)

    # Caculate model's performance
    metrics = show_model_performance(model, tokenizer, df_test)
    print(metrics)

    print('Completed.')

if __name__ == '__main__':
    main()