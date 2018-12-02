from trainautoencoder import trainautoencoder

num_epochs = 1500
trainautoencoder([100, 40, 15, 5], 'VH', num_epochs)
trainautoencoder([100, 70, 35, 5], 'VH', num_epochs)
trainautoencoder([100, 50, 25, 10], 'VH', num_epochs)
trainautoencoder([100, 70, 40, 10], 'VH', num_epochs)
trainautoencoder([200, 70, 20, 5], 'VH', num_epochs)
trainautoencoder([200, 130, 70, 5], 'VH', num_epochs)
trainautoencoder([200, 80, 30, 10], 'VH', num_epochs)
trainautoencoder([200, 140, 70, 10], 'VH', num_epochs)
