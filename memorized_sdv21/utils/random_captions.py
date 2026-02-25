import json
import numpy as np
from collections import defaultdict

maxvocab = 49400
prompt_size = 4
num_caps = 10
# filepath = "/fs/cml-projects/diffusion_rep/data/imagenette2-320/clean_captions.json"
# newfile = f"/fs/cml-projects/diffusion_rep/data/imagenette2-320/random_captions_{prompt_size}.json"

filepath = "/fs/cml-projects/diffusion_rep/data/laion_10k_random/laion_10k_captions.json"
newfile = f"/fs/cml-projects/diffusion_rep/data/laion_10k_random/random_captions_{prompt_size}.json"

random_prompts_dict = defaultdict(list)
count = 0
with open(filepath) as data_file:    
    data = json.load(data_file)
    for k,v in data.items():
        count+=1
        for n in range(num_caps):
            # import ipdb; ipdb.set_trace()
            random_prompts_dict[k].append(str(list(np.random.randint(maxvocab, size=prompt_size))))
        if count%1000==0:
            print(count)
# import ipdb; ipdb.set_trace()
with open(newfile, 'w') as fp:
    json.dump(random_prompts_dict, fp)
