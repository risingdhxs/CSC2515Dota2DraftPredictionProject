import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('a', type=str) # log file path
parser.add_argument('b', type=str) # output file name
parser.add_argument('c', type=str) # graph title



args = parser.parse_args()


with open(os.path.abspath(args.a),'r') as f:
    data = [l for l in f.readlines()]

data = [e.replace('/',' ').replace(':',' ').replace(',',' ').split() for e in data[:-2]] # the last two lines does not count for epochs and accuracies

data = np.array(data)[:,[1,5,8]].astype(float)


plt.plot(data[:,0], data[:,1], 'r', label='training')
plt.plot(data[:,0], data[:,2], 'g', label='validation')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title(args.c+('\ntraining: %f' % np.max(data[:,1]))+(', validation:%f' % np.max(data[:,2])))
plt.savefig(os.path.join(os.path.abspath('./'),args.b)+'.png')

# python draw.py fc_1.batch.log fc_1.batch.log single_layer_with_no_dropout
# python draw.py fc_2.batch.log fc_2.batch.log single_layer_with_dropout
# python draw.py fc_3.batch.log fc_3.batch.log double_layer_with_no_dropout
# python draw.py fc_4.batch.log fc_4.batch.log double_layer_with_dropout
# python draw.py fc_5.batch.log fc_5.batch.log triple_layer_with_no_dropout
# python draw.py fc_6.batch.log fc_6.batch.log triple_layer_with_dropout