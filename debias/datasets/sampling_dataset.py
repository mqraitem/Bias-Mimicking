import torch
import torch.utils.data
import random
from scipy.optimize import linprog
import numpy as np

class SamplingDataset(): 

    def get_targets_bin(self): 
        num_targets = len(torch.unique(self.targets))
        targets_ = torch.zeros((len(self.targets), num_targets)) 
        targets_[torch.arange((len(self.targets,))), self.targets] = 1 
        return targets_ 

    def set_dro_info(self):
        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets)) 

        self.groups_idx = torch.zeros((len(self.targets)))
        for i, t, b in zip(torch.arange((len(self.targets))), self.targets, self.bias_targets): 
            idx = t + (b*num_targets) 
            self.groups_idx[i] = idx 

        self.n_groups = num_targets*num_biases

    def group_counts(self): 
        counts = torch.zeros((self.n_groups, ))
        for i in range(self.n_groups): 
            counts[i] = torch.sum(self.groups_idx == i) 
        return counts

    def calculate_bias_weights(self): 
        
        groups_counts = self.group_counts()

        groups_counts = groups_counts
        groups_counts = groups_counts/torch.sum(groups_counts)         
        groups_counts = 1/groups_counts 
       
        self.group_weights = groups_counts[self.groups_idx.long()]
        self.group_weights = torch.Tensor(self.group_weights).float()   
    
    def solve_linear_program(self, target_distro, target_prime_distro): 
        num_biases = len(torch.unique(self.bias_targets)) 
        obj = [-1] * num_biases
        
        lhs_ineq = []
        for bias in range(num_biases): 
            ineq = [0] * num_biases
            ineq[bias] = 1 
            lhs_ineq.append(ineq)
        
        rhs_ineq = target_prime_distro

        lhs_eq = []
        target_distro = [x/sum(target_distro) for x in target_distro]
        for prob, bias in zip(target_distro, range(num_biases - 1)): 
            eq = [-prob]*num_biases
            eq[bias] = 1 - prob
            lhs_eq.append(eq)
        
        rhs_eq = [0]*(num_biases - 1)

        bnd = [(0, float("inf")) for _ in range(num_biases)]

        opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                      A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
                      method="revised simplex")
        
        sol = opt.x
        sol = [int(x) for x in sol]
        sol = [x if x > 0 else 1 for x in sol]
        return sol

    def get_target_distro(self, target): 
        num_biases = len(torch.unique(self.bias_targets)) 
        target_distro = [] 
        for bias in range(num_biases): 
            target_distro.append(torch.sum(torch.logical_and(self.targets == target,
                                                             self.bias_targets == bias)))
        
        return target_distro
        
    def get_kept_indices(self, target, target_prime, target_prime_new_distro):

        to_keep_indices = [] 
        for bias, bias_distro in enumerate(target_prime_new_distro):
            tmp = torch.logical_and(self.targets == target_prime, self.bias_targets == bias)
            indices_bias = list(torch.arange(len(self.targets))[tmp].numpy())
            to_keep_indices.extend(random.sample(indices_bias, bias_distro))
        
        return to_keep_indices

    def bias_mimick(self): 
        
        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets)) 

        for target in range(num_targets): 
            target_distro = self.get_target_distro(target)
            to_keep_indices = [] 
            for target_prime in range(num_targets): 
                
                if target_prime == target: 
                    indices_target = list(torch.arange(len(self.targets))[self.targets == target] )
                    to_keep_indices.extend(indices_target)
                else: 
                    target_prime_distro = self.get_target_distro(target_prime)
                    target_prime_new_distro = self.solve_linear_program(target_distro, target_prime_distro)
                    to_keep_indices.extend(self.get_kept_indices(target, target_prime, target_prime_new_distro))
            
            full_idxs = torch.arange((len(self.targets)))
            to_select = torch.ones((len(self.targets)))
            to_select[to_keep_indices] = 0 
            full_idxs = full_idxs[to_select.bool()] 

            self.targets_bin[full_idxs, target] = -1

    def over_sample_ce(self): 

        group_counts = self.group_counts() 
        max_count = int(torch.max(group_counts).item())
        to_keep_idx = [] 
        for group_idx in range(len(group_counts)):
            subsampled_indices = torch.arange(len(self.targets))[self.groups_idx==group_idx]
            subsampled_indices = list(subsampled_indices.numpy())
            subsampled_indices = random.choices(subsampled_indices, k=max_count)
            to_keep_idx.extend(subsampled_indices)

        self.set_to_keep(to_keep_idx)


    def under_sample_ce(self): 
        
        group_counts = self.group_counts() 
        min_count = int(torch.min(group_counts).item())
        to_keep_idx = [] 
        for group_idx in range(len(group_counts)):
            indices = torch.logical_and(self.groups_idx==group_idx, self.samples_check == 0)
            
            if torch.sum(indices) < min_count: 
                self.samples_check[self.groups_idx == group_idx] = 0 
                indices = torch.logical_and(self.groups_idx==group_idx, self.samples_check == 0)

            subsampled_indices = torch.arange(len(self.targets))[indices]
            subsampled_indices = list(subsampled_indices.numpy())
            subsampled_indices = random.sample(subsampled_indices, min_count)
            self.samples_check[subsampled_indices] = 1
            to_keep_idx.extend(subsampled_indices)
        
        self.set_to_keep(to_keep_idx)

    def get_targets_bin_distro(self, target_bin, num_targets, num_biases): 
        
        distro = [] 
        for target in range(num_targets):
            target_distro = []  
            for bias in range(num_biases):
                count = torch.logical_and(self.targets == target, self.bias_targets == bias)
                count = torch.logical_and(count, self.targets_bin[:, target_bin] != -1)
                target_distro.append(torch.sum(count))
            distro.append(target_distro)

        return distro

    def print_new_distro(self): 

        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets)) 

        print('===================================')
        print("Binary Labels Distribution: ")
        for target_idx in range(num_targets):
            
            print(f'Binary Target {target_idx}')
            print('---------------------------')
            target_distro = self.get_targets_bin_distro(target_idx, num_targets, num_biases)
            for target, distro in enumerate(target_distro): 
                print(f"Target {target}: {[x.item() for x in distro]}") 

        print('===================================')
        print("Normal Label Distribution: ")
        for target in range(num_targets):
            target_distro = self.get_target_distro(target)
            print(f"Target {target}: {[x.item() for x in target_distro]}")

        print('===================================')

    def get_eye_tsr(self): 

        num_targets = len(torch.unique(self.targets))
        num_biases = len(torch.unique(self.bias_targets)) 

        eye_tsr = torch.zeros((num_targets, num_biases))

        for target in range(num_targets):
            target_distro = self.get_target_distro(target)
            eye_tsr[target, np.argmin(target_distro)] = 1 

        return eye_tsr
