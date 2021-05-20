import csv
import argparse
import sys
import os
import spacy

parser = argparse.ArgumentParser(description = "Test for Word types")
parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

def main():
    parsed = parser.parse_known_args(sys.argv)
    input_file = os.path.join(parsed[0].data_dir, "train.tsv")
    output_file = os.path.join(parsed[0].output_dir, "train2.tsv")

    with open(input_file, "r", encoding="utf-8") as reader:
        tsv_reader = csv.reader(reader, delimiter="\t")
        for row in tsv_reader:
            print(row[0])

if __name__ == "__main__":
    main()
