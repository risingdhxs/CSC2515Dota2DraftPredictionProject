def trainautoencoder_legacy_1(layer, set):
    import numpy as np
    import time
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from autoencoder_def_flex import autoencoder

    batch_size = 8192
    learning_rate = 1e-3

    datapath = '../../data/all/all_IO_noleave_' + set + '.npz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_print = 100

    model = autoencoder(layer).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print('Loading ' + datapath + ' on ' + device.type)
    dataset = np.load(datapath)
    TrainX = np.asarray(np.asmatrix(dataset['TrainX'])[0, 0].astype(np.float32).todense())
    ValidX = np.asarray(np.asmatrix(dataset['ValidX'])[0, 0].astype(np.float32).todense())

    # print('Converting dataset matrices to torch tensors...')
    n = TrainX.shape[1]

    TrainXae = torch.from_numpy(np.vstack((TrainX[:, :int(n / 2)], TrainX[:, int(n / 2):]))).to(device)
    del TrainX
    dataloader_train = DataLoader(TrainXae, batch_size=batch_size, shuffle=True)

    ValidXae = torch.from_numpy(np.vstack((ValidX[:, :int(n / 2)], ValidX[:, int(n / 2):]))).to(device)
    del ValidX
    dataloader_valid = DataLoader(ValidXae, batch_size=batch_size, shuffle=False)

    if len(layer) == 3:
        print('Training AutoEncoder of layer {}-{}-{}'.format(layer[0], layer[1], layer[2]))
    else:
        print('Training AutoEncoder of layer {}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3]))

    print('Training Set size:{}, batch size:{}'.format(TrainXae.shape[0], batch_size))

    loss_opt = 1
    epoch_opt = 0
    model_state_dict_opt = model.state_dict()
    optimizer_state_dict_opt = optimizer.state_dict()

    notoverfitting = True
    epochs = 0
    loss_train = np.zeros((0, 1))
    loss_valid = np.zeros((0, 1))

    while notoverfitting:
        start = time.time()
        i = 0
        loss_sum = 0
        for data in dataloader_train:
            i += 1
            _, recon = model(data)
            loss = criterion(recon, data)
            loss_sum = loss_sum + loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        loss_train = np.vstack((loss_train, loss_sum/i))

        i = 0
        loss_sum = 0
        for data in dataloader_valid:
            i += 1
            _, recon = model(data)
            loss_sum = loss_sum + criterion(recon, data).item()
        loss_valid = np.vstack((loss_valid, loss_sum/i))

        end = time.time()

        if loss_valid[-1, 0] < loss_opt:
            loss_opt = loss_valid[-1, 0]
            epoch_opt = epochs
            model_state_dict_opt = model.state_dict()
            optimizer_state_dict_opt = optimizer.state_dict()

        if epochs % n_print == 0:
            print(
                'Finished epoch {}, training time {:.2f}s. Training loss:{:.4f}, Validation loss:{:.4f}'.format(
                    epochs, end - start, loss_train[-1, 0], loss_valid[-1, 0]))
        # ############################
        # For test purpose only
        # if epochs > 2:
        #     notoverfitting = False
        # ############################
        if epochs > 2:
            if ((loss_valid[-1, 0] > 1.2*loss_opt) & (loss_valid[-2, 0] > 1.2*loss_opt)):
                notoverfitting = False

        epochs = epochs + 1

    print('Training overfits at epoch {}; optimal model occurred at epoch {}'.format(epochs - 1, epoch_opt))
    TestX = np.asarray(np.asmatrix(dataset['TestX'])[0, 0].astype(np.float32).todense())
    TestXae = torch.from_numpy(np.vstack((TestX[:, :int(n / 2)], TestX[:, int(n / 2):]))).to(device)
    del TestX
    dataloader_test = DataLoader(TestXae, batch_size=batch_size, shuffle=False)

    model_opt = autoencoder(layer).to(device)
    model_opt.load_state_dict(model_state_dict_opt)

    i = 0
    loss_test = 0
    for data in dataloader_test:
        i += 1
        _, recon = model_opt(data)
        loss_test = loss_test + criterion(recon, data).item()

    loss_test = loss_test / i
    if len(layer) == 3:
        filetitle = 'AutoEncoder_{}_{}_{}_all_IO_noleave_{}.pt'.format(layer[0], layer[1], layer[2], set)
    else:
        filetitle = 'AutoEncoder_{}_{}_{}_{}_all_IO_noleave_{}.pt'.format(layer[0], layer[1], layer[2], layer[3], set)

    print('Finished Training {}: optimal Validation loss:{:.4f} at epoch:{}; Test Set loss:{:.4f}'.format(filetitle, loss_opt, epoch_opt, loss_test))

    torch.save({
        'model_state_dict': model_state_dict_opt,
        'optimizer_state_dict': optimizer_state_dict_opt,
        'layer': layer,
        'epochs': epochs,
        'epoch_optimal': epoch_opt,
        'loss_optimal': loss_opt,
        'loss_train': loss_train,
        'loss_valid': loss_valid,
        'loss_test': loss_test,
    }, filetitle)

