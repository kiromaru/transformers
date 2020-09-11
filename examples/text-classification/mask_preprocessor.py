import sys
import os
import csv
import torch
import argparse
import random
import time
from transformers import AutoTokenizer, AutoModelWithLMHead

parser = argparse.ArgumentParser(description = 'Dataset preprocessor using masked words')
parser.add_argument('--percent', type=float, required=False, help='Percentage of words that will be masked', default=0.15)
parser.add_argument('--task_name', type=str, required=True, help='Task name')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
#parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory')
parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')
parser.add_argument('--max_seq_length', type=int, required=True, help='Maxium sequence length')
parser.add_argument('--verbosity', type=str, required=False, choices=[ 'normal', 'verbose' ], default='normal', help='Defines verbosity')

verbosity_level = 'normal'

def is_verbose():
    if verbosity_level == 'verbose':
        return True
    return False

def get_input_filename(task: str):
    task = task.upper()

    if task == 'MNLI':
        return 'train.tsv'
    elif task == 'MNLI-MM':
        return 'train.tsv'
    else:
        raise RuntimeError(f'Unrecognized task: {task}')

def get_output_filename(task: str):
    return 'train_preprocessed.tsv'

def get_columns_to_process(task: str):
    task = task.upper()

    if task == 'MNLI':
        return [8, 9]
    elif task == 'MNLI-MM':
        return [8, 9]
    else:
        raise RuntimeError(f'Unrecognized task: {task}')

def get_sentences(row: list, input_columns: list):
    if len(input_columns) == 1:
        start = input_columns[0]
        end = start + 1
    elif len(input_columns) == 2:
        start = input_columns[0]
        end = input_columns[1] + 1
    else:
        raise RuntimeError(f'input_columns has invalid length: {len(input_columns)}')

    return row[start:end]

def replace_words(sentence, words_to_mask: int, special_tokens: list, mask_token_id: int):
    total_range = len(sentence[0])
    replaced = 0

    while replaced < words_to_mask:
        index = random.randrange(total_range)
        if sentence[0][index] not in special_tokens:
            sentence[0][index] = mask_token_id
            replaced += 1

def reconstruct_sentences(tokens: list, predicted_indexes: list, special_tokens: list, tokenizer):
    prediction_index = 0
    for i in range(len(tokens[0])):
        if tokens[0][i] == tokenizer.mask_token_id:
            tokens[0][i] = predicted_indexes[prediction_index]
            prediction_index = prediction_index + 1

    tokens_list = [ elem for elem in tokens[0] ]
    separator_idx = tokens_list.index(tokenizer.sep_token_id)
    tokens_list1 = tokens_list[:separator_idx]
    tokens_list2 = tokens_list[separator_idx + 1 :]
    tokens_list_clean1 = [ elem for elem in tokens_list1 if elem not in special_tokens ]
    tokens_list_clean2 = [ elem for elem in tokens_list2 if elem not in special_tokens ]
    sentences = []
    sentences.append(tokenizer.decode(tokens_list_clean1))
    sentences.append(tokenizer.decode(tokens_list_clean2))
    return sentences

def main():
    script_start = time.time()

    parsed = parser.parse_known_args(sys.argv)
    parsed_args = parsed[0]

    verbosity_level = parsed_args.verbosity

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    input_filename = os.path.join(parsed_args.data_dir, get_input_filename(parsed_args.task_name))
    output_filename = os.path.join(parsed_args.data_dir, get_output_filename(parsed_args.task_name))
    input_columns = get_columns_to_process(parsed_args.task_name)

    print (f'Loading model: {parsed_args.model_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(parsed_args.model_name_or_path, cache_dir=parsed_args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(parsed_args.model_name_or_path, cache_dir=parsed_args.cache_dir)
    special_tokens = [ tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id ]

    model.to(dev)

    print (f'Start processing file: {input_filename}')

    start_time = time.time()
    line_number = 0
    total_loss = 0.0

    with open(input_filename, 'r', encoding='utf8') as reader:
        with open(output_filename, 'w', encoding='utf8', newline='') as writer:
            tsv_reader = csv.reader(reader, delimiter="\t", quoting=csv.QUOTE_NONE)
            tsv_writer = csv.writer(writer, delimiter="\t")

            for row in tsv_reader:
                line_number = line_number + 1
                if line_number == 1:
                    # Write header
                    row.append("loss")
                    tsv_writer.writerow(row)
                    continue

                # Determine how many words will be replaced
                sentences = get_sentences(row, input_columns)
                compound_sentence = sentences[0] + tokenizer.sep_token + sentences[1]
                if is_verbose():
                    print(f'Compound sentence: {compound_sentence}')

                input = tokenizer.encode(compound_sentence, return_tensors='pt')
                input_copy = input.clone()
                sentence_length = len(input[0])
                
                if sentence_length > model.config.max_position_embeddings:
                    input_copy = torch.zeros(1, model.config.max_position_embeddings, dtype=input.dtype)
                    input_copy[0] = input[0][:model.config.max_position_embeddings]
                    input = input_copy.clone()
                    print (f'Line {line_number} contains sentence of length {sentence_length}, truncating to {model.config.max_position_embeddings}')

                # Replace words with mask token
                input_list = input[0].tolist()
                special_token_list = [ elem for elem in input_list if elem in special_tokens ]
                word_count = len(input[0]) - len(special_token_list)
                words_to_mask = int(round(parsed_args.percent * word_count))
                replace_words(input, words_to_mask, special_tokens, tokenizer.mask_token_id)
                mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

                if is_verbose():
                    print (f'Sentences with masks: {tokenizer.decode(input[0])}')

                # Predict words using model
                line_loss = float(0)

                model_result = model(input.to(dev), labels=input_copy.to(dev))
                line_loss = model_result[0].item()
                total_loss += line_loss
                token_logits = model_result[1]
                mask_token_logits = token_logits[0, mask_token_index, :]
                predicted_indexes = torch.argmax(mask_token_logits, dim=1)
                sentences = reconstruct_sentences(input, predicted_indexes, special_tokens, tokenizer)

                if is_verbose():
                    print (f'Loss: {line_loss}')
                    print (f'Predicted sentence 1: {sentences[0]}')
                    print (f'Predicted sentence 2: {sentences[1]}')

                row[input_columns[0]] = sentences[0]
                row[input_columns[1]] = sentences[1]
                row.append(str(line_loss))
                tsv_writer.writerow(row)

                if line_number % 100 == 0:
                    end_time = time.time()
                    print (f'Processed {line_number} lines. Lines per second: {100 / (end_time - start_time)}')
                    start_time = time.time()

    script_end = time.time()
    print ('')
    elapsed = script_end - script_start
    hours = int(elapsed / (60 * 60))
    minutes = int((elapsed - (hours * 60 * 60)) / 60)
    seconds = int(elapsed - (hours * 60 * 60) - (minutes * 60))
    print (f'Processed {line_number} lines. Total loss: {total_loss} Total elapsed time: {hours}:{minutes}:{seconds}')

if __name__ == "__main__":
    main()
