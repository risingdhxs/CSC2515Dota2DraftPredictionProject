from trainautoencoder_113 import trainautoencoder_113

set = 'VH'
L = [[130, 150, 180],
     [110, 90, 80, 70],
     [120, 150, 120, 90]]
m = len(L)
for i in range(m):
    print('Training AutoEncoder {} on set {}: {}/{}, Part 2'.format(L[i], set, i + 1, m))
    trainautoencoder_113(L[i], set)

print('Finished Training AutoEncoder of layer {} on set {}, Part 2'.format(len(L[0]), set))
