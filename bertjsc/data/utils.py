import pandas as pd
from tqdm import tqdm

def read_jwtd(path: str):
    """
    Description:
        Read jwtd(v1) to a pd.DataFrame
    """
    df = pd.read_json(path, orient='records', lines=True)
    return df

def get_ml_trainable4jwtd(df: pd.DataFrame, tokenizer, categories: list, max_unmatch: int):
    """
    Description:
        Select the trainable data from jwtd(v1) dataset for Masked-Language-Model.
    Parameters:
        df: pd.DataFrame
            Raw dataset(`train.jsonl` or `test.jsonl`)
        tokenizer: BertJapaneseTokenizer
        categories: list[str]
            The error types to be extracted('substitution', 'deletion', 'insertion', 'kanji-conversion')
        max_unmatch: int
            Max unmatch key number.
    Example:
        `df = get_ml_trainable4jwtd(df_train, ["kanji-conversion"], 3)`
    """
    trainable_df = pd.DataFrame()
    trainable_index = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['category'] in categories:
            pre_encodings = tokenizer(row['pre_text'])
            post_encodings = tokenizer(row['post_text'])
            
            # Input_ids & output_ids should be the same length for Masked-Language-Model
            if len(pre_encodings['input_ids']) == len(post_encodings['input_ids']):
                common_keys_num = sum(len(set(i))==1 for i in zip(pre_encodings['input_ids'], post_encodings['input_ids']))
                
                if len(pre_encodings['input_ids']) - common_keys_num < max_unmatch:
                    trainable_index.append(index)
    
    trainable_df = df.iloc[trainable_index]
    trainable_df = trainable_df.reset_index(drop=True)
    
    return trainable_df