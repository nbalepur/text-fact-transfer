import datasets

out_dir = '' # output directory
ds_dir = '' # dataset directory
modqga_dir = '' # directory to transferred ModQGA entities

ds = datasets.load_dataset(ds_dir)

import pickle
with open(modqga_dir, 'rb') as handle:
    data = pickle.load(handle)['test']

templates = ds['test']['template']
template_topics = ds['test']['template_title']
topics = ds['test']['title']

out = []
for i in range(len(templates)):
    curr_out = templates[i].lower()
    curr_out = curr_out.lower().replace(template_topics[i].lower(), topics[i].lower())
    
    try:
        ents, q_old, q_new, ans = data[i]

        for j, ent in enumerate(ents):
            curr_out = curr_out.replace(ent.lower().replace(",#", ", #"), ans[j].lower().replace(",#", ", #"))
    except:
        _ = 1
    out.append(curr_out)

with open(out_dir, 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)