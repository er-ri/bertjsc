import torch
import torch.nn.functional as F

# Characters ignored by tokenizer.
SKIP_LIST = [' ', '\n', '\r', '\t', '\u3000']

# Characters interpreted as '[UNK]' by tokenizer.
UNKNOWN_LIST = ['￥', '`']

# A weird phenomena, tokenizer will also interpret the following characters as a whole '[UNK]'
# when it's in a consecutive characters of UNKNOWN_LIST.
# Like:
#   '`]`' -> '[UNK]'
#   '``'  -> '[UNK]'
INSIDE_LIST = ['#', ';', '^', '[', ']', '-']

def predict_of_json(model, tokenizer, device, text):
    r"""
    Description:
        Correct the `text` and return a json form result.
    Args:
        model: Masked Language Model
        device: 'cpu' or 'gpu'
        tokenizer: Japanese tokenizer
        text: Text to check
    Return:
        JSON(key: `token`(`correct` if error found) and `scores`).
    """
    model.to(device)
    encodings = tokenizer.encode_plus(text=text, 
                        max_length=512,
                        truncation=True,
                        add_special_tokens=True, 
                        return_tensors="pt"
                ).to(device)
    token_ids = encodings['input_ids'][0]

    with torch.no_grad():
        outputs = model(**encodings)
    token_logits = outputs.logits

    scores, predict_ids = torch.topk(F.softmax(token_logits[0], dim=1), 1)
    
    # List: original token_ids, predicted token_ids and prediction scores.
    predict_index = 0
    predict_list = list(zip(token_ids.tolist()[1:-1], predict_ids.tolist()[1:-1], scores.tolist()[1:-1]))
    
    # Dicts for returning.
    result_index = 0
    result_dicts = dict()
    
    # Initialize pointer, process from the beginning of the given text.
    pointer=0
    while pointer<=len(text):
        
        if text[pointer: pointer+1] in SKIP_LIST:
            # Bypass the character ignored by tokenizer.
            result_dict = {"token": text[pointer: pointer+1], "score": float(1.0)}
            result_dicts[result_index] = result_dict
            result_index += 1
            offset = 1
        elif text[pointer: pointer+1] in UNKNOWN_LIST:
            # Handle [UNK] characters.
            unknown_length = get_unknown_token_length(text[pointer:])
            score = float("{:.6f}".format(predict_list[predict_index][2][0]))
            for i in range(unknown_length):
                result_dict = {"token": text[pointer: pointer+1], "score": score}
                pointer += 1
                result_dicts[result_index] = result_dict
                result_index += 1
            offset = 0
            predict_index += 1
        else:
            if predict_index < len(predict_list):
                original_token = handle_subwords(tokenizer.decode([predict_list[predict_index][0]]))
                predict_token = handle_subwords(tokenizer.decode([predict_list[predict_index][1][0]]))
                score = float("{:.6f}".format(predict_list[predict_index][2][0]))
                predict_index += 1

                # Construct result for the current token.
                if original_token == predict_token:
                    result_dict = {"token": original_token, "score": score}
                else:
                    result_dict = {"token": original_token, "correct": predict_token, "score": score}

                offset = len(original_token)
                result_dicts[result_index] = result_dict
                result_index += 1

        pointer += offset
    
    return result_dicts

def get_unknown_token_length(text):
    r"""
    Description:
        Get the unknown token length from the beginning of the text.
    Args:
        text: str
    Return:
        length of characters: int
    Examples:
        "\`\`\`123" return 3, while "\`\`\`]\`\`\`0000" will return 7.
    """
    pointer = 0
    while pointer<len(text):
        # 'INSIDE_LIST' should also be considered, check out the instruction.
        if text[pointer: pointer+1] in UNKNOWN_LIST or text[pointer: pointer+1] in INSIDE_LIST:
            pointer += 1
        else:
            break
    return pointer

def handle_subwords(token):
    r"""
    Description:
        Get rid of subwords '##'.
    About tokenizer subwords:
        See: https://huggingface.co/docs/transformers/tokenizer_summary
    """
    if len(token) > 2 and token[0:2] == '##':
        token = token[2:]
    return token