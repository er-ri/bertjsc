# BERTJSC
Japanese Spelling Error Corrector using BERT(Masked-Language Model).

## Abstract
The project, fine-tuned the Masked-Language BERT for the task of Japanese Spelling Error Correction. The whole word masking pretrained [model](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) has been applied in the project. Besides the original BERT, another architecture based on BERT used a method called [soft-masking](https://arxiv.org/abs/2005.07421) was also implemented in this project. The training dataset(Japanese Wikipedia Typo Dataset version 1) contains a pair of grammatical error and error-free sentence which is collected from wikipedia, can be downloaded at [here](https://nlp.ist.i.kyoto-u.ac.jp/EN/edit.php?JWTD).

## Getting Started
1. Clone the project and install necessary packages.
```
    pip install -r requirements.txt
```
2. Download the [train dataset](https://nlp.ist.i.kyoto-u.ac.jp/EN/edit.php?JWTD) and put it to `./dataset/jwtd_v1.0/`.
3. Run `python scripts/trainer.py`. The file describes the steps for the training process and includes a function to calculate model's performance metrics.
4. After the training, invoke the model by the following code.
```python
    result = predict_of_json(model, tokenizer, device, "日本語校正してみす。")
    print(result)
```
* Where `device` is the computing device. The following block shows the prediction result.
```json
{
    0: {'token': '日本語', 'score': 0.999341},
    1: {'token': '校', 'score': 0.996382},
    2: {'token': '正', 'score': 0.997387},
    3: {'token': 'し', 'score': 0.999978},
    4: {'token': 'て', 'score': 0.999999},
    5: {'token': 'み', 'score': 0.999947},
    6: {'token': 'す', 'correct': 'ます', 'score': 0.972711},
    7: {'token': '。', 'score': 1.0}
}
```

## Evaluation
### Detection
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Pre-train   |   43.5   |   31.9   |   41.5   |   36.1   |
| Fine-tune   |   77.3   | **71.1** |   81.2   | **75.8** |
| Soft-masked | **78.4** |   65.3   | **88.4** |   75.1   |

### Correction
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Pre-train   |   37.0   |   19.0   |   29.8   |   23.2   |
| Fine-tune   |   74.9   | **66.4** |   80.1   | **72.6** |
| Soft-masked | **76.4** |   61.4   | **87.8** |   72.2   |
---
* Training Platform: *AWS SageMaker Lab*
* Batch Size: *32*
* Epoch: *6*
* Learning Rate: *2e-6*
* Coefficient(soft-masked): *0.8*

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contribution
Any suggestions for improvement or contribution to this project are appreciated! Feel free to submit an issue or pull request!

## References
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
3. [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
4. [HuggingFace🤗: BERT base Japanese](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)
5. [pycorrector](https://github.com/shibing624/pycorrector)
6. [基于BERT的文本纠错模型](https://github.com/gitabtion/BertBasedCorrectionModels)
7. [SoftMasked Bert文本纠错模型实现](https://github.com/quantum00549/SoftMaskedBert)

