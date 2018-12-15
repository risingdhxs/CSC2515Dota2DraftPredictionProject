from trainautoencoder_227 import trainautoencoder_227

set = 'N'
L = [[200, 150],
     [200, 175, 150],
     [300, 150]]
m = len(L)
for i in range(m):
    print('Training SemiAutoEncoder {} on set {}: {}/{}, Part 1'.format(L[i], set, i + 1, m))
    trainautoencoder_227(L[i], set, 2)

print('Finished Training SemiAutoEncoder of layer {} on set {}, Part 1'.format(len(L[0]), set))
