import argparse
import os
import sys
import csv
import json

parser = argparse.ArgumentParser(description = 'gen-debiased-nli JSONL file processor')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--filename', type=str, required=True, help='File to read')
parser.add_argument('--sentence1', type=str, required=True, help='Name of the first sentence')
parser.add_argument('--sentence2', type=str, required=True, help='Name of the second sentence')
parser.add_argument('--label', type=str, required=True, help='Name of the actual label')

def get_label(short_label):
    if (short_label == 2):
        return "contradiction"
    if (short_label == 0):
        return "entailment"
    if (short_label == 1):
        return "neutral"
    raise ValueError("Unknown short label: " + short_label)

def main():
    parsed = parser.parse_known_args(sys.argv)

    input_file = os.path.join(parsed[0].data_dir, parsed[0].filename)
    output_file = os.path.join(parsed[0].data_dir, "processed.tsv")
    sentence1_name = parsed[0].sentence1
    sentence2_name = parsed[0].sentence2
    label_name = parsed[0].label

    read_rows = 0
    with open(output_file, 'wt', newline='', encoding='utf-8') as tsvfileout:
        tsvwriter = csv.writer(tsvfileout, delimiter='\t')
        # Write header
        row = [ "index","promptID","pairID","genre","sentence1_binary_parse","sentence2_binary_parse","sentence1_parse","sentence2_parse","sentence1","sentence2","label","gold_label" ]
        tsvwriter.writerow(row)
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            line = " "
            while line:
                line = infile.readline()
                if line is None:
                    break
                if len(line) == 0:
                    continue

                try:
                    decoded_line = json.loads(line)
                    row[0] = read_rows #index
                    row[1] = read_rows #promptID
                    row[2] = read_rows #pairID
                    if "genre" in decoded_line:
                        row[3] = decoded_line["genre"]
                    else:
                        row[3] = "na"
                    row[4] = "s1" 
                    row[5] = "s2"
                    row[6] = "s1"
                    row[7] = "s2"
                    row[8] = decoded_line[sentence1_name].strip('\r\n')
                    row[9] = decoded_line[sentence2_name].strip('\r\n')
                    row[10] = get_label(decoded_line[label_name])
                    row[11] = get_label(decoded_line[label_name])

                    tsvwriter.writerow(row)
                    read_rows = read_rows + 1

                    if read_rows % 500 == 0:
                        print ("Read " + str(read_rows) + " rows")
                except KeyboardInterrupt:
                    print("Interrupted!")
                    break
                except:
                    e = sys.exc_info()[0]
                    print("Failed to process row: " + str(read_rows))
                    print(e)
                    print(row)

if __name__ == "__main__":
    main()
