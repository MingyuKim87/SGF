


import json
from collections import defaultdict
import numpy as np

filepath = "/fs/cml-projects/diffusion_rep/data/laion_100k_random_sdv2p1/caption.json"
newfile = f"/fs/cml-projects/diffusion_rep/data/laion_100k_random_sdv2p1/l100kaion_combined_captions.json"
random_prompts_dict = defaultdict(list)
count = 0
with open(filepath) as data_file:    
    data = json.load(data_file)
    for k,v in data.items():
        # import ipdb; ipdb.set_trace()
        count+=1
        newk = k.replace("images_large", "train/images_large")
        random_prompts_dict[newk].append(v)
        if count%5000==0:
            print(count)
# import ipdb; ipdb.set_trace()
with open(newfile, 'w') as fp:
    json.dump(random_prompts_dict, fp)


# import json
# from collections import defaultdict
# import numpy as np

# ogcaps = f"/fs/cml-projects/diffusion_rep/data/laion_10k_random_aesthetics_5plus/laion_aesthetics_10k_captions.json"
# blipcaps = f"/fs/cml-projects/diffusion_rep/data/laion_10k_random_aesthetics_5plus/laion_aesthetics_10k_blip_captions.json"

# newfile = f"/fs/cml-projects/diffusion_rep/data/laion_10k_random_aesthetics_5plus/laion_aesthetics_combined_captions.json"

# with open(blipcaps) as data_file:
#     blipdata = json.load(data_file)

# combined_prompts_dict = defaultdict(list)
# count = 0
# with open(ogcaps) as data_file:    
#     data = json.load(data_file)
#     for k,v in data.items():
#         count+=1
#         blipvs = blipdata[k]
#         combined_prompts_dict[k] = v+blipvs
#         if count%1000==0:
#             print(count)
# # import ipdb; ipdb.set_trace()
# with open(newfile, 'w') as fp:
#     json.dump(combined_prompts_dict, fp)