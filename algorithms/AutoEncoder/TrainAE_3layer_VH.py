from trainautoencoder import trainautoencoder

set = 'VH'
L = [[200, 50, 10],
     [200, 100, 10],
     [200, 150, 10],
     [200, 55, 15],
     [200, 110, 15],
     [200, 165, 15],
     [200, 65, 20],
     [200, 110, 20],
     [200, 155, 20]]
m = len(L)
for i in range(m):
    print('Training {} layer AutoEncoder on set {}: {}/{}'.format(len(L[0]), set, i + 1, m))
    trainautoencoder(L[i], set)

print('Finished Training AutoEncoder of layer {} on set {}'.format(len(L[0]), set))