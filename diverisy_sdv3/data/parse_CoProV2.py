import json
import pandas as pd
import random

# 1. Load JSONL data line by line
data = []
with open('datasets/CoProv2/metadata.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e} in line: {line}")

# 2. Rename 'caption' key to 'prompt'
for item in data:
    if 'caption' in item:
        item['prompt'] = item.pop('caption')
    else:
        item['prompt'] = ''

# 3. Remove duplicate prompts
unique_prompt_dict = {}
for item in data:
    prompt = item['prompt']
    if prompt not in unique_prompt_dict:
        unique_prompt_dict[prompt] = item  # Keep only the first occurrence

# 4. Convert dictionary to a list
unique_prompt_data = list(unique_prompt_dict.values())
print(f"📌 Number of unique prompts: {len(unique_prompt_data)}")

# 5. Randomly sample up to 10,000 items
sample_size = min(10000, len(unique_prompt_data))
sampled_data = random.sample(unique_prompt_data, sample_size)

# 6. Convert to DataFrame and add case_number column
df = pd.DataFrame(sampled_data)
df.insert(0, 'case_number', range(1, len(df) + 1))  # Add 'case_number' starting from 1

# 7. Save as CSV
df.to_csv('CoProV2_unique_10k.csv', index=False)