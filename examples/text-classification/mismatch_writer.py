import csv
import argparse
import sys
import os

parser = argparse.ArgumentParser(description = 'Mismatch writer for Bert')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--task_name', type=str, required=True, help='Task name')

def main():
    parsed = parser.parse_known_args(sys.argv)

    file_suffix = ''
    input_file = ''
    output_file_suffix = ''

    # Read the input data file mismatched output file
    task_name =  parsed[0].task_name.upper()

    if (task_name == 'MNLI'):
        file_suffix = 'mnli'
        input_file = os.path.join(parsed[0].data_dir, 'dev_matched.tsv')
    elif (task_name == 'MNLI-MM'):
        file_suffix = 'mnli-mm'
        input_file = os.path.join(parsed[0].data_dir, 'dev_mismatched.tsv')
        output_file_suffix = '-mm'
    elif (task_name == 'MNLI-HANS'):
        file_suffix = 'mnli-hans'
        input_file = os.path.join(parsed[0].data_dir, 'dev_matched.tsv')
    elif (task_name == 'MNLI-HANS-MM'):
        file_suffix = 'mnli-hans-mm'
        input_file = os.path.join(parsed[0].data_dir, 'dev_mismatched.tsv')
        output_file_suffix = '-mm'
    elif (task_name == 'QQP'):
        file_suffix = 'qqp'
        input_file = os.path.join(parsed[0].data_dir, 'dev.tsv')
    elif (task_name == 'MRPC'):
        file_suffix = 'mrpc'
        input_file = os.path.join(parsed[0].data_dir, 'dev.tsv')
    else:
        raise RuntimeError('Unknown task name')           

    data = []
    with open(input_file, 'r', encoding='utf-8') as tsvfile:
        data = tsvfile.readlines()

    input_file = os.path.join(parsed[0].output_dir, f"mismatched_results_{file_suffix}.txt")
    output_file = os.path.join(parsed[0].output_dir, f"mismatched_data{output_file_suffix}.tsv")

    with open(input_file, 'r') as reader:
        with open(output_file, 'w', encoding='utf-8') as writer:
            # Include the header
            writer.write(data[0])

            # Now the actual data
            line = reader.readline()
            while (line is not None and line is not ''):
                idx = int(line)
                idx = idx + 1
                if (idx >= len(data)):
                    raise RuntimeError('Index should not be bigger than input file size')
                writer.write(data[idx])
                line = reader.readline()

if __name__ == "__main__":
    main()
