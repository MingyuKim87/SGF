import pandas as pd

# Load the CSV file
df = pd.read_csv('datasets/i2p_top_10p_prompts.csv')  # Replace 'your_file.csv' with the actual file name or path

# Calculate the number of words in each prompt
df['prompt_word_count'] = df['prompt'].apply(lambda x: len(str(x).split()))

# Calculate the number of characters in each prompt
df['prompt_char_count'] = df['prompt'].apply(lambda x: len(str(x)))

# Calculate statistics for word count
average_word_count = df['prompt_word_count'].mean()
std_word_count = df['prompt_word_count'].std()

# Calculate statistics for character count
average_char_count = df['prompt_char_count'].mean()
std_char_count = df['prompt_char_count'].std()

# Print the results
print(f"The average number of words in the prompts is {average_word_count:.2f} words.")
print(f"The standard deviation of word counts is {std_word_count:.2f} words.\n")

print(f"The average number of characters in the prompts is {average_char_count:.2f} characters.")
print(f"The standard deviation of character counts is {std_char_count:.2f} characters.")