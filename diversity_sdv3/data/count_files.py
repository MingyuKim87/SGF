import os

directory = 'results/all/negation_003/std_SD_v3v1-4_CoProV2/all'  # 예: 'datasets/CoProv2'
file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

print(f"Number of files in '{directory}': {file_count}")