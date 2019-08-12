import pickle
import glob
import numpy as np
import ipdb

mode = ['strong', 'weak']
fnames = sorted(glob.glob('./rate_of_convergence/' + mode[0] + '*'))

table = np.zeros([3, 5, 2])

ipdb.set_trace()

for i in range(5):
    for j in range(3):
        f = open(fnames[3*i + j], 'rb')
        tmp = pickle.load(f)
        f.close()
        table[j, i, :] = tmp['params']

f = open('rate_of_conv_table.txt', 'wb')
pickle.dump(table, f)
f.close()
