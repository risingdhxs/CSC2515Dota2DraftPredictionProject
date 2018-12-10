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
plt.ylabel('loss')
plt.title(args.c+('\ntraining: %f' % np.min(data[:,1]))+(', validation:%f' % np.min(data[:,2]))+(', best validation: epoch %d' % np.argmin(data[:,2])))
plt.savefig(os.path.join(os.path.abspath('./'),args.b)+'.loss.png')
