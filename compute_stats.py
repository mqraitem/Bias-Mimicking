import os
import numpy as np

import sys 

dir_ = sys.argv[1]
methods = os.listdir(dir_) 

for method in methods: 

    print('%s results:'%method)
    method_dir = os.path.join(dir_, method) 
    result_files = os.listdir(method_dir)
    

    results = {} 
    for result_file in result_files: 
        data = open(os.path.join(method_dir, result_file), 'r') 
        result_line = data.readlines()[-4].split(',')
        for line in result_line: 
            if 'best_valid_test' in line: 
                metric = line.split(':')[0].split('/')[1]
                res = line.split(':')[1].strip().replace('}', '').replace("'", '')
                if metric in results: 
                    results[metric].append(float(res)) 
                else: 
                    results[metric] = [float(res)]
        
        data.close() 
    for metric in results: 
        print(metric, ': %.1f, %.1f'%(np.mean(results[metric]), np.std(results[metric]))) 

    print('-------------------------------------------')
