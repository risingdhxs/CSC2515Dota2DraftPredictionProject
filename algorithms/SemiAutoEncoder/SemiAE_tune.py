from trainautoencoder_227_tune import trainautoencoder_227
import numpy as np

set = 'N_small'
rl = [0.5, 1, 2, 5, 10]
L = [[200, 50],
     [200, 100],
     [200, 150],
     [200, 200],
     [200, 250],
     [200, 300],
     [200, 100, 50],
     [200, 150, 100],
     [200, 175, 150],
     [200, 200, 200],
     [200, 225, 250],
     [200, 250, 300],
     [300, 100],
     [300, 150],
     [300, 200],
     [300, 250],
     [300, 300],
     [300, 200, 100],
     [300, 225, 150],
     [300, 250, 200],
     [300, 275, 250],
     [300, 300, 300]]
m = len(L)
n = 5
decoderACC = np.zeros((m, len(rl)))
encodeLRACC = np.zeros((m, len(rl)))
for i in range(m):
    for j in range(len(rl)):
        for k in range(n):
            deACC, enACC = trainautoencoder_227(L[i], set, rl[j])
            if deACC > decoderACC[i, j]:
                decoderACC[i, j] = deACC
            if enACC > encodeLRACC[i, j]:
                encodeLRACC[i, j] = enACC

print('Finished Semi-AE parameter tuning')
print('decoder Accuracy')
print(decoderACC)
print('encode LR Accuracy')
print(encodeLRACC)
