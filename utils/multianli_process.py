import argparse
import os
import sys
import csv
import json

parser = argparse.ArgumentParser(description = 'JSONL file processor')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--filename', type=str, required=True, help='File to read')

def main():
    parsed = parser.parse_known_args(sys.argv)

    input_file = os.path.join(parsed[0].data_dir, parsed[0].filename)
    output_file = os.path.join(parsed[0].data_dir, "processed.tsv")

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
                    row[1] = decoded_line["promptID"]
                    row[2] = decoded_line["pairID"]
                    row[3] = decoded_line["genre"]
                    row[4] = decoded_line["sentence1_binary_parse"].strip('\r\n')
                    row[5] = decoded_line["sentence2_binary_parse"].strip('\r\n')
                    row[6] = decoded_line["sentence1_parse"].strip('\r\n')
                    row[7] = decoded_line["sentence2_parse"].strip('\r\n')
                    row[8] = decoded_line["sentence1"].strip('\r\n')
                    row[9] = decoded_line["sentence2"].strip('\r\n')
                    row[10] = decoded_line["gold_label"]
                    row[11] = decoded_line["gold_label"]

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
