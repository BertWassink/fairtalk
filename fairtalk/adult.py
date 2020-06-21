import numpy as np

with open('./data/adult.data') as f:
    d = f.read() #reads the entire file

print(d.shape)