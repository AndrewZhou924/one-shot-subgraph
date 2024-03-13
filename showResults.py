import os
import argparse
import pickle as pkl
import numpy as np

'''
usage (WN18RR as an example):
python3 showResults.py --file ./results/WN18RR/search_log.pkl
'''

parser = argparse.ArgumentParser(description="Parser for the decouple framework")
parser.add_argument('--file', type=str, default='')
parser.add_argument('--topk', type=int, default=10)
args = parser.parse_args()
data = pkl.load(open(args.file, 'rb'))

# search mode
if type(data) == dict and 'Namespace' in list(data.keys())[0]:
    HP_key_list = []
    val_mrr_list, test_mrr_list = [], []
    val_eval_dict_list, test_eval_dict_list = [], []
    params_list = []
    for HP_key, HP_values in data.items():
        (best_mrr, best_test_mrr, params, opts) = HP_values
        HP_key_list.append(HP_key)
        val_mrr_list.append(best_mrr)
        test_mrr_list.append(best_test_mrr)
        params_list.append(params)
else:
    exit()
        
print(f'==> finish {len(val_mrr_list)} exp already.')
print(f'==> the top-{args.topk} resulst:')

# show top-k config
if len(val_mrr_list) > 0:
    for idx in np.argsort(np.array(val_mrr_list))[::-1][:args.topk]:
        print('*'*50)
        print('val_mrr:', val_mrr_list[idx], 'HPs:',params_list[idx])
    print('*'*50)

exit()