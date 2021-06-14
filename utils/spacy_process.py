import argparse
import spacy
import sys
import os
import csv

parser = argparse.ArgumentParser(description = 'Spacy processor')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--filename', type=str, required=True, help='File to read')
parser.add_argument('--sentence1', type=int, required=True, help='Index of sentence 1 to process')
parser.add_argument('--sentence2', type=int, required=True, help='Index of sentence 2 to process')

nlp = spacy.load("en_core_web_sm")

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def build_tagged_sentence(sentence):
    doc = nlp(sentence)
    tagged_sentence = " "
    tagged_words = []
    for token in doc:
        if not token.is_punct:
            tagged_words.append(token.tag_)
        tagged_words.append(token.text)

    return tagged_sentence.join(tagged_words)

def main():
    parsed = parser.parse_known_args(sys.argv)

    input_file = os.path.join(parsed[0].data_dir, parsed[0].filename)
    output_file = os.path.join(parsed[0].data_dir, "processed.tsv")
    sentence1idx = parsed[0].sentence1
    sentence2idx = parsed[0].sentence2

    print ("Reading input file: " + input_file)

    read_rows = 0
    first = True    
    with open(input_file, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        with open(output_file, 'wt', newline='', encoding='utf-8') as tsvfileout:
            tsvwriter = csv.writer(tsvfileout, delimiter='\t')
            for row in tsvreader:
                if first:
                    first = False
                    tsvwriter.writerow(row)
                    continue

                read_rows = read_rows + 1

                try:
                    # Original
                    # tsvwriter.writerow(row)

                    # Tagged
                    sentence = row[sentence1idx]
                    row[sentence1idx] = build_tagged_sentence(sentence)
                    sentence = row[sentence2idx]
                    row[sentence2idx] = build_tagged_sentence(sentence)
                    tsvwriter.writerow(row)

                    if read_rows % 500 == 0:
                        print ("Read " + str(read_rows) + " rows")
                except KeyboardInterrupt:
                    print("Interrupted!")
                    break
                except:
                    e = sys.exc_info()[0]
                    print ("Failed to process row: " + str(read_rows))
                    print (e)
                    print (row)

    print ("Read " + str(read_rows) + " rows")

if __name__ == "__main__":
    main()
