import sys
import argparse
import os
import mask_preprocessor as mp
import run_glue as rg

parser = argparse.ArgumentParser(description = 'Process to mask input file and finetune with hugging face')
parser.add_argument('--num_train_epochs', type=int, required=True, help='Number of training epochs to execute')
parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')


def replace_parameter_value(param_name: str, param_value: str):
    for i in range(len(sys.argv)):
        if sys.argv[i] == param_name:
            sys.argv[i + 1] = param_value
            return

def main():
    parsed = parser.parse_known_args(sys.argv)
    parsed_args = parsed[0]

    output_dir = parsed_args.output_dir
    num_epochs = parsed_args.num_train_epochs

    # Finetuning will execute one epoch at a time
    replace_parameter_value('--num_train_epochs', '1.0')

    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    for epoch in range(num_epochs):
        iter_output = os.path.join(output_dir, 'iter' + str(epoch))

        # Remove any cache for training data
        files = os.listdir(parsed_args.cache_dir)
        for f in files:
            if f.startswith('cached_train'):
                cache_file = os.path.join(parsed_args.cache_dir, f)
                os.remove(cache_file)

        replace_parameter_value('--output_dir', iter_output)

        # First step: Mask words, predict and calculate individual loss
        mp.main()

        # Second step: Fine tune with hugging face
        rg.main()

        # Model for next iteration is the result of this epoch
        replace_parameter_value('--model_name_or_path', iter_output)

if __name__ == "__main__":
    main()
