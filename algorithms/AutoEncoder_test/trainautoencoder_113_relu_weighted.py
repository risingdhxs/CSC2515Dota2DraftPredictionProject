def trainautoencoder_113_relu(layer, set):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import time
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from autoencoder_113_relu import autoencoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import copy

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
    m_train = TrainX.shape[0]
    ValidX = np.asarray(np.asmatrix(dataset['ValidX'])[0, 0].astype(np.float32).todense())
    m_valid = ValidX.shape[0]
    TestX = np.asarray(np.asmatrix(dataset['TestX'])[0, 0].astype(np.float32).todense())
    m_test = TestX.shape[0]

    # w_train = np.ones(TrainX.shape)
    # l_train = np.where(TrainX)
    # w_train[l_train[0], l_train[1]] = 21.6
    # w_valid = np.ones(ValidX.shape)
    # l_valid = np.where(ValidX)
    # w_valid[l_valid[0], l_valid[1]] = 21.6
    # w_test = np.ones(TestX.shape)
    # l_test = np.where(TestX)
    # w_test[l_test[0], l_test[1]] = 21.6

        # print('Converting dataset matrices to torch tensors...')
    n = TrainX.shape[1]

    TrainXae = torch.from_numpy(np.vstack((TrainX[:, :int(n / 2)], TrainX[:, int(n / 2):]))).to(device)
    # del TrainX
    dataloader_train = DataLoader(TrainXae, batch_size=batch_size, shuffle=True)

    ValidXae = torch.from_numpy(np.vstack((ValidX[:, :int(n / 2)], ValidX[:, int(n / 2):]))).to(device)
    # del ValidX
    dataloader_valid = DataLoader(ValidXae, batch_size=batch_size, shuffle=False)

    TestXae = torch.from_numpy(np.vstack((TestX[:, :int(n / 2)], TestX[:, int(n / 2):]))).to(device)
    # del TestX

    if len(layer) == 3:
        nodenum = '{}-{}-{}'.format(layer[0], layer[1], layer[2])
    elif len(layer) == 4:
        nodenum = '{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3])
    elif len(layer) == 6:
        nodenum = '{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5])
    elif len(layer) == 10:
        nodenum = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5], layer[6], layer[7], layer[8], layer[9])

    print('Training ReLu AutoEncoder of {}. Training set size:{}, batch size:{}'.format(nodenum, TrainXae.shape[0], batch_size))

    loss_opt = 1
    epoch_opt = 0
    model_state_dict_opt = model.state_dict()

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
            w = np.ones(data.shape).astype(np.float32)
            lw = np.where(data)
            w[lw[0],lw[1]] = 21.6
            wt = torch.from_numpy(w).to(device)
            criterion.weight = wt
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
            w = np.ones(data.shape).astype(np.float32)
            lw = np.where(data)
            w[lw[0], lw[1]] = 21.6
            wt = torch.from_numpy(w).to(device)
            criterion.weight = wt
            loss_sum = loss_sum + criterion(recon, data).item()
        loss_valid = np.vstack((loss_valid, loss_sum/i))

        end = time.time()

        if loss_valid[-1, 0] < loss_opt:
            loss_opt = loss_valid[-1, 0]
            epoch_opt = epochs
            model_state_dict_opt = copy.deepcopy(model.state_dict())

        if epochs % n_print == 0:
            print(
                'Finished epoch {}, time {:.2f}s. Training loss:{:.4f}, Validation loss:{:.4f}'.format(
                    epochs, end - start, loss_train[-1, 0], loss_valid[-1, 0]))
            print('Comparing Reconstructed Draft with Actual Draft')
            recon = recon.cpu().data.numpy()
            l1 = np.where(data[0, :])
            print('heros {}'.format(l1[0]))
            print('Empty Positions:')
            print(recon[0, l1[0] + 2])
            print(data[0, l1[0] + 2])
            print('Hero Positions:')
            print(recon[0, l1])
            print(data[0, l1])

        ############################
        # For test purpose only: short termination
        # if epochs > 2:
        #     notoverfitting = False
        ############################
        # run at least 100 epoch after the optimal model
        if epochs > 100 + epoch_opt:
            if loss_valid[-1, 0] > 1.2*loss_opt:
                notoverfitting = False

        if epochs > 2000:
            notoverfitting = False

        epochs = epochs + 1

    print('Training overfits at epoch {}; optimal model occurred at epoch {} with validation loss {:.4f}'.format(
        epochs - 1, epoch_opt, loss_opt))

    # Commented out for testing purpose
    # x = np.arange(0, epochs)
    # fig = plt.figure()
    # plt.plot(x, loss_train, 'g', x, loss_valid, 'b')
    # plt.legend(('Training Loss', 'Validation Loss'))
    # plt.title('AE of layer {} on {} data, optimal validation loss {:.4f}'.format(nodenum, set, loss_opt))
    # plt.xlabel('Training Epochs')
    # plt.ylabel('Reconstruction Cross Entropy Loss')
    # plt.grid(True)
    # fig.savefig('AutoEncoder_{}_all_IO_noleave_{}.jpg'.format(nodenum, set))

    print('Encoding Train/Valid/Test sets...')

    dataloader_train = DataLoader(TrainXae, batch_size=batch_size, shuffle=False)
    dataloader_valid = DataLoader(ValidXae, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(TestXae, batch_size=batch_size, shuffle=False)

    encode_train = np.zeros((0, layer[-1]))
    encode_valid = np.zeros((0, layer[-1]))
    encode_test = np.zeros((0, layer[-1]))
    recon_train = np.zeros((0, int(n / 2)))
    recon_valid = np.zeros((0, int(n / 2)))
    recon_test = np.zeros((0, int(n / 2)))

    model_opt = autoencoder(layer).to(device)
    model_opt.load_state_dict(model_state_dict_opt)

    i = 0
    loss_train = 0
    for data in dataloader_train:
        i += 1
        encode, recon = model_opt(data)
        w = np.ones(data.shape).astype(np.float32)
        lw = np.where(data)
        w[lw[0], lw[1]] = 21.6
        wt = torch.from_numpy(w).to(device)
        criterion.weight = wt
        encode_train = np.vstack((encode_train, encode.cpu().data.numpy()))
        recon_train = np.vstack((recon_train, recon.cpu().data.numpy()))
        loss_train = loss_train + criterion(recon, data).item()
    encode_train = np.hstack((encode_train[:m_train, :], encode_train[m_train:, :]))
    recon_train = np.hstack((recon_train[:m_train, :], recon_train[m_train:, :]))
    loss_train = loss_train / i

    i = 0
    loss_valid = 0
    for data in dataloader_valid:
        i += 1
        encode, recon = model_opt(data)
        w = np.ones(data.shape).astype(np.float32)
        lw = np.where(data)
        w[lw[0], lw[1]] = 21.6
        wt = torch.from_numpy(w).to(device)
        criterion.weight = wt
        encode_valid = np.vstack((encode_valid, encode.cpu().data.numpy()))
        recon_valid = np.vstack((recon_valid, recon.cpu().data.numpy()))
        loss_valid = loss_valid + criterion(recon, data).item()
    encode_valid = np.hstack((encode_valid[:m_valid, :], encode_valid[m_valid:, :]))
    recon_valid = np.hstack((recon_valid[:m_valid, :], recon_valid[m_valid:, :]))
    loss_valid = loss_valid / i

    i = 0
    loss_test = 0
    for data in dataloader_test:
        i += 1
        encode, recon = model_opt(data)
        w = np.ones(data.shape).astype(np.float32)
        lw = np.where(data)
        w[lw[0], lw[1]] = 21.6
        wt = torch.from_numpy(w).to(device)
        criterion.weight = wt
        encode_test = np.vstack((encode_test,  encode.cpu().data.numpy()))
        recon_test = np.vstack((recon_test, recon.cpu().data.numpy()))
        loss_test = loss_test + criterion(recon, data).item()
    encode_test = np.hstack((encode_test[:m_test, :], encode_test[m_test:, :]))
    recon_test = np.hstack((recon_test[:m_test, :], recon_test[m_test:, :]))
    loss_test = loss_test / i

    print('On {} dataset, optimal {} AutoEncoder T/V/T loss: {:.4f}/{:.4f}/{:.4f}'.format(set, nodenum, loss_train,
                                                                                          loss_valid, loss_test))

    TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
    ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
    TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())

    print('Comparing Reconstructed Draft with Actual Draft')
    g1=1
    g2=10

    l1 = np.where(TrainX[g1, :])
    print('Game {}, heros {}'.format(g1 + 1, l1[0]))
    print('Empty Positions:')
    print(recon_train[g1, l1[0] + 2])
    print(TrainX[g1, l1[0] + 2])
    print('Hero Positions:')
    print(recon_train[g1, l1])
    print(TrainX[g1, l1])

    l2 = np.where(TrainX[g2, :])
    print('Game {}, heros {}'.format(g2 + 1, l2[0]))
    print('Empty Positions:')
    print(recon_train[g2, l2[0] + 2])
    print(TrainX[g2, l2[0] + 2])
    print('Hero Positions:')
    print(recon_train[g2, l2])
    print(TrainX[g2, l2])

    lr = LogisticRegression(random_state=0, solver='newton-cg')
    lr.fit(encode_train, TrainY.ravel())
    score_train_lr = lr.score(encode_train, TrainY.ravel())
    score_valid_lr = lr.score(encode_valid, ValidY.ravel())
    score_test_lr = lr.score(encode_test, TestY.ravel())

    print('Encoding Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr,
                                                                                 100 * score_valid_lr,
                                                                                 100 * score_test_lr))

    gauss_nb = GaussianNB()
    gauss_nb.fit(encode_train, TrainY.ravel())
    score_train_nb = gauss_nb.score(encode_train, TrainY.ravel())
    score_valid_nb = gauss_nb.score(encode_valid, ValidY.ravel())
    score_test_nb = gauss_nb.score(encode_test, TestY.ravel())

    print('Encoding Gaussian Naive Bayes Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb,
                                                                                  100 * score_valid_nb,
                                                                                  100 * score_test_nb))

    lr_recon = LogisticRegression(random_state=0, solver='newton-cg')
    lr_recon.fit(recon_train, TrainY.ravel())
    score_train_lr_recon = lr_recon.score(recon_train, TrainY.ravel())
    score_valid_lr_recon = lr_recon.score(recon_valid, ValidY.ravel())
    score_test_lr_recon = lr_recon.score(recon_test, TestY.ravel())

    print('Recon LR Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_recon,
                                                                      100 * score_valid_lr_recon,
                                                                      100 * score_test_lr_recon))

    gauss_nb_recon = GaussianNB()
    gauss_nb_recon.fit(recon_train, TrainY.ravel())
    score_train_nb_recon = gauss_nb_recon.score(recon_train, TrainY.ravel())
    score_valid_nb_recon = gauss_nb_recon.score(recon_valid, ValidY.ravel())
    score_test_nb_recon = gauss_nb_recon.score(recon_test, TestY.ravel())

    print('Recon NB Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb_recon,
                                                                      100 * score_valid_nb_recon,
                                                                      100 * score_test_nb_recon))

    lr_2 = LogisticRegression(random_state=0, solver='newton-cg')
    lr_2.fit(TrainX, TrainY.ravel())
    score_train_lr_2 = lr_2.score(TrainX, TrainY.ravel())
    score_valid_lr_2 = lr_2.score(ValidX, ValidY.ravel())
    score_test_lr_2 = lr_2.score(TestX, TestY.ravel())

    print('Raw Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_2,
                                                                                     100 * score_valid_lr_2,
                                                                                     100 * score_test_lr_2))

    gauss_nb_2 = GaussianNB()
    gauss_nb_2.fit(TrainX, TrainY.ravel())
    score_train_nb_2 = gauss_nb_2.score(TrainX, TrainY.ravel())
    score_valid_nb_2 = gauss_nb_2.score(ValidX, ValidY.ravel())
    score_test_nb_2 = gauss_nb_2.score(TestX, TestY.ravel())

    print('Raw Gaussian Naive Bayes Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb_2,
                                                                                      100 * score_valid_nb_2,
                                                                                      100 * score_test_nb_2))

    print('On {} dataset, optimal {} ReLu AE T/V/T loss: {:.4f}/{:.4f}/{:.4f}'.format(set, nodenum, loss_train,
                                                                                          loss_valid, loss_test))

    # filetitle = 'AutoEncoder_{}_all_IO_noleave_{}.pt'.format(nodenum, set)
    #
    # torch.save({
    #     'model_state_dict': model_state_dict_opt,
    #     'layer': layer,
    #     'epochs': epochs,
    #     'epoch_optimal': epoch_opt,
    #     'loss_optimal': loss_opt,
    #     'loss_train': loss_train,
    #     'loss_valid': loss_valid,
    #     'loss_test': loss_test,
    #     'score_train_lr': score_train_lr,
    #     'score_valid_lr': score_valid_lr,
    #     'score_test_lr': score_test_lr,
    #     # 'score_train_svm': score_train_svm,
    #     # 'score_valid_svm': score_valid_svm,
    #     # 'score_test_svm': score_test_svm,
    #     'score_train_nb': score_train_nb,
    #     'score_valid_nb': score_valid_nb,
    #     'score_test_nb': score_test_nb,
    # }, filetitle)
