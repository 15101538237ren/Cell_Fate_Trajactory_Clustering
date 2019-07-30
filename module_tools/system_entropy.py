import pandas as pd
import numpy as np
import os


def calc_system_entropy(probvec):
    probvec = np.absolute(probvec)
    system_entropy = np.real(-np.sum(np.multiply(probvec, np.log(probvec))))
    return system_entropy

if __name__ == "__main__":
    print('Script file to calculated system entropy "svalue" from ProbVec')