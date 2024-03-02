import os
import pandas as pd
import tiktoken

# Function to count the number of tokens using tiktoken
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to count tokens in all CSV files in a directory
def count_tokens_in_directory(directory_path):
    total_tokens = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            total_tokens += count_tokens_in_file(file_path)
    return total_tokens

# Function to count tokens in a single CSV file
def count_tokens_in_file(file_path):
    df = pd.read_csv(file_path)
    df['combined_text'] = df['article_text_Ngram']  # Replace with your text columns
    token_counts = df['combined_text'].apply(num_tokens_from_string)
    return token_counts.sum()

# Path to the directory containing CSV files
directory_path = '/Users/kev/Downloads/now'

# Count tokens in the directory
# total_token_count = count_tokens_in_directory(directory_path)
total_token_count = count_tokens_in_file('/Users/kev/Downloads/now/TZ_domestic_Ngram_stopword_lematize.csv')
print(f"The total number of tokens in the directory is: {total_token_count}")
