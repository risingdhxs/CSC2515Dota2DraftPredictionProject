from trainautoencoder_227 import trainautoencoder_227

set = 'N'
L = [[300, 175, 150],
     [300, 200],
     [300, 250, 200]]
m = len(L)
for i in range(m):
    print('Training SemiAutoEncoder {} on set {}: {}/{}, Part 2'.format(L[i], set, i + 1, m))
    trainautoencoder_227(L[i], set, 2)

print('Finished Training SemiAutoEncoder of layer {} on set {}, Part 2'.format(len(L[0]), set))
