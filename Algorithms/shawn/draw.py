import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('a', type=int) # log file path
parser.add_argument('b', type=int) # output file name



args = parser.parse_args()


with open(os.path.abspath(args.a),'r') as f:
    data = [l for l in f.readlines()]

data = [e.replace('/',' ').replace(':',' ').replace(',',' ').split() for e in data]

data = np.array(data)[:,[1,5,8]].astype(float)


plt.plot(data[:,0], data[:,1], 'r', label='training')
plt.plot(data[:,0], data[:,2], 'g', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Single layer FC')
plt.savefig(os.path.join(os.path.abspath(args.a),args.b)+'.png')