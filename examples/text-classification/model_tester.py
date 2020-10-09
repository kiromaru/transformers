import sys
import os
import csv
import torch
import argparse
import random
import time
import transformers
import collections
#from transformers import AutoTokenizer, AutoModelWithLMHead

parser = argparse.ArgumentParser(description = 'Dataset preprocessor using masked words')
parser.add_argument('--percent', type=float, required=False, help='Percentage of words that will be masked', default=0.15)
parser.add_argument('--task_name', type=str, required=True, help='Task name')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory')
parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')
parser.add_argument('--verbosity', type=str, required=False, choices=[ 'normal', 'verbose' ], default='normal', help='Defines verbosity')

verbosity_level = 'normal'
line_modulo = 1000


def main():
    script_start = time.time()

    model_filepath = "F:\\repos\\glue_data\\MNLI\\outputForBertTest\\checkpoint-10000"
    model_filename = os.path.join(model_filepath, "pytorch_model.bin")
    cache_dir = "f:\\repos\\cache"
    prediction_layers = [
        'cls.predictions.bias',
        'cls.predictions.transform.dense.weight',
        'cls.predictions.transform.dense.bias',
        'cls.predictions.transform.LayerNorm.weight',
        'cls.predictions.transform.LayerNorm.bias',
        'cls.predictions.decoder.weight',
        'cls.predictions.decoder.bias'
    ]
    classifier_layers = [
        'classifier.weight',
        'classifier.bias'
    ]

    print (f'Loading model: {model_filepath}')
    bert_config = transformers.BertConfig()
    #new_model = transformers.BertForMaskedLM(bert_config)
    new_model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    model_dict = new_model.state_dict()
    partial_dict = collections.OrderedDict()
    for layer in prediction_layers:
        partial_dict[layer] = model_dict[layer]

    partial_model_name = os.path.join(model_filepath, "pytorch_model_partial.bin")
    torch.save(partial_dict, partial_model_name)

    new_model2 = transformers.BertForMaskedLM(bert_config)
    model_dict = torch.load(model_filename)
    for layer in classifier_layers:
        model_dict.pop(layer, None)
    
    loaded_partial_weights = torch.load(partial_model_name)
    for layer in loaded_partial_weights.keys():
        model_dict[layer] = loaded_partial_weights[layer]

    new_model2.load_state_dict(model_dict)

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)

    test_sentence = "This is the sentence I want to predict. Wonder if it will work"
    tokenized = tokenizer.encode(test_sentence, return_tensors='pt')
    tokenized[0][10] = tokenizer.mask_token_id
    mask_token_index = torch.where(tokenized == tokenizer.mask_token_id)[1]

    model_result = new_model2(tokenized)
    token_logits = model_result[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    predicted_indexes = torch.argmax(mask_token_logits, dim=1)

    print(f"Masked sentence: {tokenizer.decode(tokenized[0])}")
    print(f"Predicted word: {tokenizer.decode(predicted_indexes)}")


    # tokenizer = transformers.BertTokenizer()
    # tokenizer = AutoTokenizer.from_pretrained(model_filepath, cache_dir=cache_dir)
    # model = AutoModelWithLMHead.from_pretrained(model_filepath, cache_dir=cache_dir)
    # special_tokens = [ tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id ]

    # print (f'Start processing file: {input_filename}')




if __name__ == "__main__":
    main()
