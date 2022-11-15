import mapping_accuracy
import numpy as np

def code_to_vec(p, code):
    def char_to_vec(c):
        y = np.zeros((len(mapping_accuracy.CHARS),))
        y[mapping_accuracy.CHARS.index(c)] = 1.0
        return y

    c = np.vstack([char_to_vec(c) for c in code])

    return np.concatenate([[1. if p else 0], c.flatten()])