import csv
import argparse
import sys
import os

parser = argparse.ArgumentParser(description = 'Mismatch writer for Bert')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

def main():
    parsed = parser.parse_known_args(sys.argv)

    input_file = os.path.join(parsed[0].data_dir, 'heuristics_evaluation_set.txt')

    print ("Reading input file: " + input_file)
    data = []
    with open(input_file, 'r') as tsvfile:
        data = tsvfile.readlines()
    
    mismatch_file = os.path.join(parsed[0].output_dir, 'hans_mismatched.txt')
    output_file = os.path.join(parsed[0].data_dir, 'heuristics_evaluation_mismatched.txt')

    print("Reading mismatch file: " + mismatch_file)
    print("Writing output file: " + output_file)

    written_lines = 0
    with open(output_file, "w") as writer:
        # Write header
        writer.write(data[0])

        with open(mismatch_file, "r") as reader:
            line = reader.readline()
            while line != None and line != '':
                idx = int(line)
                idx = idx + 1
                if (idx >= len(data)):
                    raise RuntimeError('Index out of range of input file')
                writer.write(data[idx])
                written_lines = written_lines + 1
                line = reader.readline()

    print("Wrote " + str(written_lines) + " lines.")

if __name__ == "__main__":
    main()
