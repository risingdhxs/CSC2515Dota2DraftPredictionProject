from trainautoencoder_113_relu_weighted import trainautoencoder_113_relu
set = 'N'

L = [[130, 150, 120],
     [130, 150, 90],
     [130, 150, 180],
     [130, 150, 120, 90],
     [100, 80, 100, 130],
     [130, 150, 120, 90, 70, 50]]
m = len(L)
for i in range(m):
    print('Training {} layer weighted AutoEncoder on set {}: {}/{}'.format(len(L[i]), set, i + 1, m))
    trainautoencoder_113_relu(L[i], set)

print('Finished Training Weighted AutoEncoder of layer {} on set {}'.format(len(L[0]), set))
