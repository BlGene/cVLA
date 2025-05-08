# define my own LogitsProcessor
import numpy as np
from scipy.signal import argrelextrema
import torch
import transformers

def maxima_ind(probas, n=1):
    """
    probas: 1D float array
    n: int, number of points to be checked before and after

    return: indices of maxima, sorted by decreasing proba
    """
    max_ind = argrelextrema(probas, np.greater_equal, order=n)[0]
    # Achtung! Suppress close points with same value
    res = [max_ind[0]]
    for ind in max_ind:
        if ind - res[-1] > n:
            res.append(ind)
    max_ind = res

    val_ind = [(probas[i], i) for i in max_ind] # list of pairs (value, index)
    val_ind = sorted(val_ind, reverse=True)
    ind = [i for (_, i) in val_ind]
    return ind

def softmax(p, T=1):
    """
    p: 1D float array
    T: float, temperature
    """
    p = np.exp(p / T) + 1e-6
    return p / np.sum(p)

def maxima_batch(scores, n=1, top_k=3):
    """
    Compute local maxima for batch input.
    Strategies:
    1) Leave top k local maxima, deterministic
    2) Leave 1 from top k using sampling.
       Depends on softmax temperature.
    Suppress all other tokens.
    
    scores: np.ndarray (batch_size, vocab_size)
    n: int, number of points to be checked before and after
    top_k: int, number of local maxima to leave

    return: np.ndarray (batch_size, vocab_size)
    """
    batch_size, vocab_size = scores.shape
    scores = scores.cpu().numpy()
    for batch in range(batch_size):
        ind = maxima_ind(scores[batch], n=n)
        ind = ind[:top_k]

        enable_sampling = False
        if enable_sampling:
            p = softmax(scores[batch][ind], T=3) # CHANGE ME: temperature
            ind = np.random.choice(ind, 1, p=p)

        # suppress all non-maxima
        is_max = np.zeros((vocab_size), dtype=int)
        is_max[ind] = 1
        scores[batch][is_max == 0] = -float('inf')

    return torch.tensor(scores)

class MyProcessor(transformers.LogitsProcessor):
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Modify the logit distribution at each prediction step.
        
        scores: logits (batch_size, vocab_size)
        """
        batch_size, vocab_size = scores.shape
        
        loc_first = vocab_size - 1024 - 128 - 64
        loc_last = vocab_size - 128 - 64

        # suppress language tokens
        scores[:, :loc_first] = -float('inf')

        # modify loc token distribution
        scores[:, loc_first:loc_last] = maxima_batch(scores[:, loc_first:loc_last], n=100)

        # IMPORTANT: modify seg
        seg_first = vocab_size - 128 - 64
        seg_last = vocab_size - 64
        scores[:, seg_first:seg_last] = maxima_batch(scores[:, seg_first:seg_last], n=100)
        
        return scores


def save_model_output(generation, filename, save_beams=False):
    """
    tokens vector = language tokens, 1024 x loc, 128 x seg, 64 x special
    """
    #if type(generation) is list:
    #    # concatenate generations into one numpy array like with beam search
    #    scores = torch.stack([gen.scores for gen in generation]).cpu().numpy() #  (n_beams, prediction_step, 1, n_tokens)
    #    scores = np.transpose(scores, (2, 1, 0, 3))[0]
    #else:
    scores = torch.stack(generation.scores).cpu().numpy() # (prediction_steps, 1, n_tokens)
    _, _, n_tokens = scores.shape
    loc_first = n_tokens - 1024 - 128 - 64
    loc_last = n_tokens - 128 - 64
    scores = scores[:, :, loc_first:loc_last]
    
    with open(filename, "wb") as f:
        np.save(f, generation.sequences.cpu().numpy()) # (n_beams, enc_steps)
        np.save(f, scores) # (prediction_steps, n_beams, n_tokens)
        if save_beams:
            np.save(f, generation.sequences_scores.cpu().numpy()) # (n_beams)