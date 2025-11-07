# Tokenizer

This script trains a tokenizer using the Byte Pair Encoding (BPE) algorithm on a given text corpus.

## Description

The `tokenizer.py` script reads text files from a specified directory, and then uses the BPE algorithm to create a tokenizer. The trained tokenizer, including the merge rules and vocabulary size, is saved to a JSON file.

### Key Features:

- **Byte Pair Encoding (BPE):**  Builds a vocabulary by iteratively merging the most frequent pairs of tokens.
- **File Handling:** Reads all `.txt` files from a specified directory.
- **Tokenizer Saving:** Saves the learned merge rules and vocabulary size to a `tokenizer.json` file.
- **Encoding and Decoding:** Includes functions to encode text into tokens and decode tokens back into text using the trained tokenizer.

## How to Use

1. **Place your corpus:**
   - Put all your `.txt` files in the `corpus` directory.

2. **Run the script:**
   ```bash
   python tokenizer.py
   ```
3. **Output:**
    - The script will create a `tokenizer.json` file in the root directory, which contains the trained tokenizer.

## Functions

- `get_file_paths(directory)`: Returns all the file names in the directory.
- `show_sample_output(file_paths)`: Shows a sample output of a file.
- `read_all_text_into_buf(file_paths, num_files=100)`: Reads all the text into a buffer.
- `encode_text(text:str, display_sample:bool= False)`: Encodes text into a list of integers.
- `get_stats(ids)`: Gets the frequency of pairs of tokens.
- `merge_pairs(ids, pair, idx)`: Merges a pair of tokens into a new token.
- `save_tokenizer(merges, vocab_size, output_file='tokenizer.json')`: Saves the merges and vocabulary size to a file.
- `encode(text, merges)`: Encodes text using the trained merges.
- `decode(ids, merges)`: Decodes a list of ids back to text.
