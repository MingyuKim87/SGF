import json
import pandas as pd
from collections import defaultdict
import random

# 1. Load the JSON
with open('datasets/CoPro/CoPro_v1.0.json', 'r') as f:
    raw_data = json.load(f)

entries = raw_data['ID_train_data']

# 2. Build DataFrame with idx
data = []
for idx, item in enumerate(entries, start=1):
    data.append({
        'idx': idx,
        'unsafe_prompt': item.get('unsafe_prompt', ''),
        'safe_prompt': item.get('safe_prompt', ''),
        'concept': item.get('concept', ''),
        'category': item.get('category', '')
    })

df = pd.DataFrame(data)

# 3. Create balanced subsamples
def create_balanced_subset(df, total_size, seed=42):
    random.seed(seed)
    grouped = df.groupby('category')
    categories = list(grouped.groups.keys())
    num_per_category = total_size // len(categories)

    subsampled = []
    for category in categories:
        group = grouped.get_group(category)
        if len(group) < num_per_category:
            raise ValueError(f"Not enough samples in category '{category}' to extract {num_per_category} items.")
        subsampled.append(group.sample(n=num_per_category, random_state=seed))

    balanced_df = pd.concat(subsampled).reset_index(drop=True)
    # balanced_df.insert(0, 'idx', range(1, len(balanced_df) + 1))  # Reassign idx
    return balanced_df

# 4. Create and save 10k and 1k datasets
df_10k = create_balanced_subset(df, total_size=10000)
df_1k = create_balanced_subset(df, total_size=1000)

df_10k.to_csv('CoPro_balanced_10k.csv', index=False)
df_1k.to_csv('CoPro_balanced_1k.csv', index=False)

print("✅ Saved balanced datasets: CoPro_balanced_10k.csv and CoPro_balanced_1k.csv")