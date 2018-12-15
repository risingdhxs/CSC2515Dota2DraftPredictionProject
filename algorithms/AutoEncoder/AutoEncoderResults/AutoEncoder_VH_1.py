from trainautoencoder_113 import trainautoencoder_113

set = 'VH'
L = [[100, 75, 50],
     [110, 90, 70],
     [130, 160, 200, 160, 120, 90]]
m = len(L)
for i in range(m):
    print('Training AutoEncoder {} on set {}: {}/{}, Part 1'.format(L[i], set, i + 1, m))
    trainautoencoder_113(L[i], set)

print('Finished Training AutoEncoder of layer {} on set {}, Part 1'.format(len(L[0]), set))
