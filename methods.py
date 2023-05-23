import numpy as np 
import os 
import pickle
from glob import glob
from scipy.stats import permutation_test

def extend_arr(arr, l): 
    """Extend given array to length l by 
    buffering with final value of array"""
        
    if len(arr) < l: 
        buffer = [arr[-1]]*(l-len(arr))
        arr.extend(buffer)
        assert len(arr) == l

    return arr


def test_by_betting(seq1, seq2, alpha=0.05): 
    """Run testing by betting algorithm with ONS betting strategy 
    on given sequences"""
    
    wealth = 1
    wealth_hist = [1] 
    const = 2 / (2 - np.log(3))
    lambd = 0 
    zt2 = 0 
    for t in range(1,min(len(seq1), len(seq2))): 
                
        St = 1 - lambd*(seq1[t] - seq2[t])
        wealth = wealth * St 
        wealth_hist.append(wealth)
        if wealth > 1/alpha: 
            return wealth_hist, 'reject'
        
        # Update lambda via ONS  
        g = seq1[t] - seq2[t]
        z = g / (1 - lambd*g)
        zt2 += z**2
        lambd = max(min(lambd - const*z/(1 + zt2), 1/2), -1/2)
        
    U = np.random.uniform()
    if wealth > U/alpha: 
        return wealth_hist, 'reject'
    
    return wealth_hist, 'sustain'

def betting_experiment(seq1, seq2, alphas, iters=10, shift_time=None): 
    """Helper to run a batch of sequential tests via betting"""
    
    results = []
    rejections = []
    s1, s2 = seq1, seq2
    for _ in range(iters): 
        if shift_time != None: 
            s1, s2 = shuffle_with_shift(seq1, seq2, shift_time)
        else: 
            np.random.shuffle(s1)
            np.random.shuffle(s2)
        
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_betting(s1, s2, alpha=alpha)
            taus.append(len(wealth))
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections

def shuffle_with_shift(seq1, seq2, shift_time): 
    """Randomly shuffle given sequences but respecting shift_time. 
    All observations before (after) shift_time remain (before) after in 
    shuffled result"""

    s1_pre, s1_post = seq1[:shift_time], seq1[shift_time:]
    s2_pre, s2_post = seq2[:shift_time], seq2[shift_time:]
    np.random.shuffle(s1_pre), np.random.shuffle(s1_post)
    np.random.shuffle(s2_pre), np.random.shuffle(s2_post)
    s1 = np.concatenate((s1_pre, s1_post))
    s2 = np.concatenate((s2_pre, s2_post))
    
    return s1, s2
        

            
def test_stat(x, y, axis):
    """Test statistic for permutation test"""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def perm_test(seq1, seq2, p):    
    """Perform permutation test"""
    
    res = permutation_test((seq1, seq2), test_stat, vectorized=True,
                           n_resamples=2000, alternative='two-sided')

    if res.pvalue <= p: 
        return True 
    return False
    
def seq_perm_test(seq1, seq2, p=0.05, k=100, bonferroni=False): 
    """Perform sequential permutation test at given significance level (p). 
    Hypothesis is tested after each k observations. 
    If indicated, perform bonferroni like correction by dividing required significance
    by 2 each batch"""
    
    l = min(len(seq1), len(seq2))
    for i in range(int(l/k)): 
        pi = p / 2**(i+1) if bonferroni else p
        if perm_test(seq1[i*k:k*(i+1)], seq2[i*k:k*(i+1)], pi): 
            # print('Reject')
            return k*(i+1), 'reject'
    return l, 'sustain'

def seq_perm_test_experiment(seq1, seq2, alphas, iters=10, k=100, bonferroni=False, shift_time=None): 
    """Helper to run batch of sequential permutation tests. k indicates window size"""
    
    results = []
    rejections = []
    s1, s2 = seq1, seq2
    for _ in range(iters): 
        taus, rejects = [], []
        if shift_time != None: 
            s1, s2 = shuffle_with_shift(seq1, seq2, shift_time)
        else: 
            np.random.shuffle(s1)
            np.random.shuffle(s2)
        for alpha in alphas: 
            steps, reject = seq_perm_test(s1, s2, p=alpha, k=k, bonferroni=bonferroni)
            taus.append(steps)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
    

def get_mean_std(arr): 
    return np.mean(arr, axis=0), np.std(arr, axis=0)

def plt_mean_std(ax, arr, alphas, label, color='navy', plot_std=False, **kwargs): 
    """Plot helper"""
    mean, std = get_mean_std(arr)
    ax.plot(alphas, mean, label=label, c=color, **kwargs)
    if plot_std: 
        ax.fill_between(alphas, mean-std/2, mean+std/2, alpha=0.05, color=color)

def save_results(name, data): 
    
    if not os.path.exists('data'): 
        os.makedirs('data')
        
    n = np.sum([x.startswith(name) for x in os.listdir('data/')])
    with open(f'data/{name}-{n}.p', 'wb') as f: 
        pickle.dump(data, f)
        

def load_results(fname): 
    results = []
    for name in glob(f'data/{fname}*'):
        with open(f'{name}', 'rb') as f: 
            content = pickle.load(f)
            results.extend(content)
    return results 

              