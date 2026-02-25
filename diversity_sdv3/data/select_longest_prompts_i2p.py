import pandas as pd

# Read the CSV file
df = pd.read_csv('datasets/i2p.csv')

# Calculate the length of each prompt and store it in a new column
df['prompt_length'] = df['prompt'].str.len()

# Find the length threshold for the top 10% longest prompts
threshold = df['prompt_length'].quantile(0.9)

# Filter prompts that are within the top 10% longest
df_top_10_percent = df[df['prompt_length'] >= threshold]

# Save the resulting DataFrame to a new CSV file
df_top_10_percent.to_csv('i2p_top_10p_prompts.csv', index=False)

# Print the total number of records and the first few records
print(f"Total records: {len(df_top_10_percent)}")
print(df_top_10_percent.head())
