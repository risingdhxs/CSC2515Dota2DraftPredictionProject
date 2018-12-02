from trainautoencoder import trainautoencoder

num_epochs = 1000
trainautoencoder([100, 25, 5], 'N', num_epochs)
trainautoencoder([100, 50, 5], 'N', num_epochs)
trainautoencoder([100, 35, 10], 'N', num_epochs)
trainautoencoder([100, 50, 10], 'N', num_epochs)
trainautoencoder([200, 30, 5], 'N', num_epochs)
trainautoencoder([200, 100, 5], 'N', num_epochs)
trainautoencoder([200, 45, 10], 'N', num_epochs)
trainautoencoder([200, 100, 10], 'N', num_epochs)