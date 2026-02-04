
import pandas as pd
import config
import os
import spacy
import scispacy
from tqdm import tqdm
import re
from collections import Counter
import string

def vocab_with_spacy(text):
    nlp = spacy.load("en_core_sci_lg")

    text = text.lower()

    medical_phrases = set()
    chunk_size = 500000
    for chunk in tqdm(chunk_text(text, chunk_size), total=int(len(text)/chunk_size), desc="Computing chunks"):
        doc = nlp(chunk)
        for ent in doc.ents:
            # Filter out numbers
            if not re.search(r'\d', ent.text):
                # Filter out stop words and punctuation
                if not all(token.is_stop or token.is_punct for token in nlp(ent.text)):
                    # Filter out proper nouns
                    if ent.label_ != "PERSON":
                        # Filter out short tokens
                        if len(ent.text) > 2:
                            medical_phrases.add(ent.text)

    return sorted(medical_phrases)

def chunk_text(text, max_chars=500000):
    length = len(text)
    for i in range(0, len(text), max_chars):
        end = min(i+max_chars, length)
        yield text[i:end]



print("Extracting reports from CheXpert dataset...")
df = pd.read_csv(os.path.join(config.DATASETS,'chexpert/df_chexpert_plus_240401.csv'))
reports = df['report'].tolist()
reports = ' '.join(reports)

# print("Extracting reports from Radlex dataset...")
# with open(os.path.join(config.VOCABS,'radlex_sentences.txt'), 'r') as f:
#     lines = f.readlines()
# reports = ' '.join(lines)

print("Extracting vocabulary from CheXpert reports...")
result = vocab_with_spacy(reports)
print("Vocabulary extraction completed.")

print("Number of unique words extracted:", len(result))
# Save the result to a txt file
print("Saving vocabulary to txt file...")
path = os.path.join(config.VOCABS,'chexpert_report_vocabulary_spacy3.txt')

with open(path, 'a') as f:
    for item in result:
        f.write(item)
        f.write('\n')

