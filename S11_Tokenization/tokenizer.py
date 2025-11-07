import os
import logging
import random
from collections import Counter

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_file_paths(directory):
    """
        Returns all the file names in directory
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
             if filename.lower().endswith('.txt'):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
    return file_paths

def show_sample_output(file_paths):
    """
        Shows sample output of a file
    """
    file_name = random.choice(file_paths)
    logger.info(f"File Name = {file_name}")
    with open(file=file_name, mode='r', encoding='utf-8') as f:
        logger.info(f.read())

def read_all_text_into_buf(file_paths, num_files=100):
    """
        Reads all the text into buffer
    """
    text = ''
    for file_path in file_paths[:num_files]:
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            text += f.read()
    return text

def encode_text(text:str, display_sample:bool= False):
    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens))
    if display_sample:
        logger.info(f"Sample tokens =\n{tokens[:20]}")
    
    return tokens

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pairs(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# Save merges for later use
def save_tokenizer(merges, vocab_size, output_file='S11_Tokenization/tokenizer.json'):
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'merges': {str(k): v for k, v in merges.items()},
            'vocab_size': vocab_size
        }, f, ensure_ascii=False)

# Encode function using trained merges
def encode(text, merges):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        tokens = merge_pairs(tokens, pair, merges[pair])
    return tokens

# Decode function
def decode(ids, merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    tokens = b''.join(vocab[idx] for idx in ids)
    return tokens.decode('utf-8', errors='replace')

def main():
    file_paths = get_file_paths('S11_Tokenization/corpus')
    # show_sample_output(file_paths)
    text = read_all_text_into_buf(file_paths)
    tokens = encode_text(text, display_sample=True)
    logger.info(f"Number of characters = {len(text)}, Number of tokens = {len(tokens)}")
    counts = Counter(tokens)
    logger.info(f"Unique bytes: {len(counts)}, Most common: {counts.most_common(10)}")

    desired_vocab_size = 256
    num_merges = desired_vocab_size - len(counts)
    ids = tokens.copy()
    
    merges = {}
    for i in range(num_merges):
        counts = get_stats(ids)
        pair = max(counts, key=counts.get)
        idx = 256 + i
        merges[pair] = idx
        print(f"merging {pair} into a new token {idx}")
        ids = merge_pairs(ids, pair, idx)
        merges[pair] = idx

        logger.info(f"Final compression: {len(tokens)/len(ids):.2f}x")
        save_tokenizer(merges, desired_vocab_size)

if __name__ == "__main__":
    main()