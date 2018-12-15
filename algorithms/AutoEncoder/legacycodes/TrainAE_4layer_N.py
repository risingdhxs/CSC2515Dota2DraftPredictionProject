from trainautoencoder import trainautoencoder

set = 'N'
L = [[200, 80, 30, 10],
     [200, 135, 70, 10],
     [200, 95, 45, 20],
     [200, 145, 75, 20],
     [200, 110, 60, 30],
     [200, 150, 90, 30]]
m = len(L)
for i in range(m):
    print('Training {} layer AutoEncoder on set {}: {}/{}'.format(len(L[i]), set, i + 1, m))
    trainautoencoder(L[i], set)

print('Finished Training AutoEncoder of {} layer on set {}'.format(len(L[0]), set))
