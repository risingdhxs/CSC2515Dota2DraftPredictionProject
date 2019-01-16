from trainautoencoder_227 import trainautoencoder_227

set = ['VH', 'N']
L = [[100, 50],
     [200, 100, 50],
     [200, 50],
     [200, 100],
     [300, 100],
     [300, 100, 50],
     [300, 150, 100],
     [300, 200, 100],
     [200, 300, 200, 100],
     [200, 300, 100, 50]]
ratio = [2, 5, 10]
m = len(L)
for i in range(len(set)):
    for j in range(len(L)):
        for k in range(len(ratio)):
            print('Training SemiAutoEncoder {} on set {} with ratio {}'.format(L[j], set[i], ratio[k]))
            trainautoencoder_227(L[j], set[i], ratio[k])

print('Finished Training SemiAutoEncoders')
