def trainautoencoder_2(layer, set):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import time
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from autoencoder_def_flex import autoencoder
    from sklearn.linear_model import LogisticRegression
    # from sklearn import svm
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
    else:
        nodenum = '{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5])

    print('Training AutoEncoder of {}. Training set size:{}, batch size:{}'.format(nodenum, TrainXae.shape[0], batch_size))

    loss_opt = 1
    epoch_opt = 0
    model_state_dict_opt = model.state_dict()
    # optimizer_state_dict_opt = optimizer.state_dict()

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
            model_state_dict_opt = copy.deepcopy(model.state_dict())
            # optimizer_state_dict_opt = coptimizer.state_dict()

        if epochs % n_print == 0:
            print(
                'Finished epoch {}, time {:.2f}s. Training loss:{:.4f}, Validation loss:{:.4f}'.format(
                    epochs, end - start, loss_train[-1, 0], loss_valid[-1, 0]))
        ############################
        # For test purpose only
        # if epochs > 2:
        #     notoverfitting = False
        ############################
        if epochs > 100:
            if ((loss_valid[-1, 0] > 1.2*loss_opt) & (loss_valid[-2, 0] > 1.2*loss_opt)):
                notoverfitting = False

        epochs = epochs + 1

    print('Training overfits at epoch {}; optimal model occurred at epoch {} with validation loss {:.4f}'.format(
        epochs - 1, epoch_opt, loss_opt))

    # x = np.arange(0, epochs)
    # fig = plt.figure()
    # plt.plot(x, loss_train, 'g', x, loss_valid, 'b')
    # plt.legend(('Training Loss', 'Validation Loss'))
    # plt.title('AE of layer {} on {} data, optimal validation loss {:.4f}'.format(nodenum, set, loss_opt))
    # plt.xlabel('Training Epochs')
    # plt.ylabel('Reconstruction Cross Entropy Loss')
    # plt.grid(True)
    # fig.savefig('AutoEncoder_{}_all_IO_noleave_{}.jpg'.format(nodenum, set))

    dataloader_train_2 = DataLoader(TrainXae, batch_size=batch_size, shuffle=False)
    dataloader_valid_2 = DataLoader(ValidXae, batch_size=batch_size, shuffle=False)
    dataloader_test_2 = DataLoader(TestXae, batch_size=batch_size, shuffle=False)

    encode_train = np.zeros((0, layer[-1]))
    encode_valid = np.zeros((0, layer[-1]))
    encode_test = np.zeros((0, layer[-1]))
    data_recon_train = np.zeros((0, int(n / 2)))
    data_recon_valid = np.zeros((0, int(n / 2)))
    data_recon_test = np.zeros((0, int(n / 2)))

    model_opt = autoencoder(layer).to(device)
    model_opt.load_state_dict(model_state_dict_opt)

    # encode_train_entire, _ = model_opt(TrainXae)
    # encode_train_entire = encode_train_entire.cpu().data.numpy()
    # encode_train_entire = np.hstack((encode_train_entire[:m_train, :], encode_train_entire[m_train:, :]))
    # encode_valid_entire, _ = model_opt(ValidXae)
    # encode_valid_entire = encode_valid_entire.cpu().data.numpy()
    # encode_valid_entire = np.hstack((encode_valid_entire[:m_valid, :], encode_valid_entire[m_valid:, :]))
    # encode_test_entire, _ = model_opt(TestXae)
    # encode_test_entire = encode_test_entire.cpu().data.numpy()
    # encode_test_entire = np.hstack((encode_test_entire[:m_test, :], encode_test_entire[m_test:, :]))

    i = 0
    loss_train = 0
    for data in dataloader_train_2:
        data_recon_train = np.vstack((data_recon_train, data))
        i += 1
        encode, recon = model_opt(data)
        encode_train = np.vstack((encode_train, encode.cpu().data.numpy()))
        loss_train = loss_train + criterion(recon, data).item()
    encode_train = np.hstack((encode_train[:m_train, :], encode_train[m_train:, :]))
    data_recon_train = np.hstack((data_recon_train[:m_train, :], data_recon_train[m_train:, :]))
    loss_train = loss_train / i

    i = 0
    loss_valid = 0
    for data in dataloader_valid_2:
        data_recon_valid = np.vstack((data_recon_valid, data))
        i += 1
        encode, recon = model_opt(data)
        encode_valid = np.vstack((encode_valid, encode.cpu().data.numpy()))
        loss_valid = loss_valid + criterion(recon, data).item()
    encode_valid = np.hstack((encode_valid[:m_valid, :], encode_valid[m_valid:, :]))
    data_recon_valid = np.hstack((data_recon_valid[:m_train, :], data_recon_valid[m_train:, :]))
    loss_valid = loss_valid / i

    i = 0
    loss_test = 0
    for data in dataloader_test_2:
        data_recon_test = np.vstack((data_recon_test, data))
        i += 1
        encode, recon = model_opt(data)
        encode_test = np.vstack((encode_test,  encode.cpu().data.numpy()))
        loss_test = loss_test + criterion(recon, data).item()
    encode_test = np.hstack((encode_test[:m_test, :], encode_test[m_test:, :]))
    data_recon_test = np.hstack((data_recon_test[:m_train, :], data_recon_test[m_train:, :]))
    loss_test = loss_test / i

    print('On {} dataset, optimal {} AutoEncoder T/V/T loss: {:.4f}/{:.4f}/{:.4f}'.format(set, nodenum, loss_train,
                                                                                          loss_valid, loss_test))

    TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
    ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
    TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())

    # print('Shape of TrainX, Encode_Train, TrainY are {}/{}/{}'.format(TrainX.shape, encode_train.shape, TrainY.shape))
    # print('Shape of ValidX, Encode_Valid, ValidY are {}/{}/{}'.format(ValidX.shape, encode_valid.shape, ValidY.shape))
    # print('Shape of TestX, Encode_Test, TestY are {}/{}/{}'.format(TestX.shape, encode_test.shape, TestY.shape))

    lr = LogisticRegression(random_state=0, solver='newton-cg')
    lr.fit(encode_train, TrainY.ravel())
    score_train_lr = lr.score(encode_train, TrainY.ravel())
    score_valid_lr = lr.score(encode_valid, ValidY.ravel())
    score_test_lr = lr.score(encode_test, TestY.ravel())

    print('Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr,
                                                                                 100 * score_valid_lr,
                                                                                 100 * score_test_lr))
    #
    # lr_entire = LogisticRegression(random_state=0, solver='newton-cg')
    # lr_entire.fit(encode_train_entire, TrainY.ravel())
    # score_train_lr_entire = lr_entire.score(encode_train_entire, TrainY.ravel())
    # score_valid_lr_entire = lr_entire.score(encode_valid_entire, ValidY.ravel())
    # score_test_lr_entire = lr_entire.score(encode_test_entire, TestY.ravel())
    #
    # print('Entire Encoding, Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_entire,
    #                                                                              100 * score_valid_lr_entire,
    #                                                                              100 * score_test_lr_entire))

    # Support Vector Machine performance has been demonstrated to be similar to LR and NB in the benchmark cases;
    # It's too slow in training... skipped here.
    # svmclf_lin = svm.LinearSVC()
    # svmclf_lin.fit(encode_train, TrainY.ravel())
    # score_train_svm = svmclf_lin.score(encode_train, TrainY)
    # score_valid_svm = svmclf_lin.score(encode_valid, ValidY)
    # score_test_svm = svmclf_lin.score(encode_test, TestY)
    #
    # print('Linear SVM Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_svm, 100 * score_valid_svm, 100 * score_test_svm))

    gauss_nb = GaussianNB()
    gauss_nb.fit(encode_train, TrainY.ravel())
    score_train_nb = gauss_nb.score(encode_train, TrainY.ravel())
    score_valid_nb = gauss_nb.score(encode_valid, ValidY.ravel())
    score_test_nb = gauss_nb.score(encode_test, TestY.ravel())

    print('Gaussian Naive Bayes Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb,
                                                                                  100 * score_valid_nb,
                                                                                  100 * score_test_nb))
    #
    # gauss_nb_entire = GaussianNB()
    # gauss_nb_entire.fit(encode_train_entire, TrainY.ravel())
    # score_train_nb_entire = gauss_nb.score(encode_train_entire, TrainY.ravel())
    # score_valid_nb_entire = gauss_nb.score(encode_valid_entire, ValidY.ravel())
    # score_test_nb_entire = gauss_nb.score(encode_test_entire, TestY.ravel())
    #
    # print('Entire Encoding, Gaussian Naive Bayes Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb_entire,
    #                                                                               100 * score_valid_nb_entire,
    #                                                                               100 * score_test_nb_entire))

    lr_2 = LogisticRegression(random_state=0, solver='newton-cg')
    lr_2.fit(TrainX, TrainY.ravel())
    score_train_lr_2 = lr_2.score(TrainX, TrainY.ravel())
    score_valid_lr_2 = lr_2.score(ValidX, ValidY.ravel())
    score_test_lr_2 = lr_2.score(TestX, TestY.ravel())

    print('Original Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_2,
                                                                                          100 * score_valid_lr_2,
                                                                                          100 * score_test_lr_2))

    lr_2_recon = LogisticRegression(random_state=0, solver='newton-cg')
    lr_2_recon.fit(data_recon_train, TrainY.ravel())
    score_train_lr_2_recon = lr_2_recon.score(data_recon_train, TrainY.ravel())
    score_valid_lr_2_recon = lr_2_recon.score(data_recon_valid, ValidY.ravel())
    score_test_lr_2_recon = lr_2_recon.score(data_recon_test, TestY.ravel())

    print('Recon Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_2_recon,
                                                                                 100 * score_valid_lr_2_recon,
                                                                                 100 * score_test_lr_2_recon))

    gauss_nb_2 = GaussianNB()
    gauss_nb_2.fit(TrainX, TrainY.ravel())
    score_train_nb_2 = gauss_nb_2.score(TrainX, TrainY.ravel())
    score_valid_nb_2 = gauss_nb_2.score(ValidX, ValidY.ravel())
    score_test_nb_2 = gauss_nb_2.score(TestX, TestY.ravel())

    print('Original Gaussian Naive Bayes Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb_2,
                                                                                  100 * score_valid_nb_2,
                                                                                  100 * score_test_nb_2))

    gauss_nb_2_recon = GaussianNB()
    gauss_nb_2_recon.fit(data_recon_train, TrainY.ravel())
    score_train_nb_2_recon = gauss_nb_2_recon.score(data_recon_train, TrainY.ravel())
    score_valid_nb_2_recon = gauss_nb_2_recon.score(data_recon_valid, ValidY.ravel())
    score_test_nb_2_recon = gauss_nb_2_recon.score(data_recon_test, TestY.ravel())

    print('Recon Gaussian Naive Bayes Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_nb_2_recon,
                                                                                           100 * score_valid_nb_2_recon,
                                                                                           100 * score_test_nb_2_recon))

    # if len(layer) == 3:
    #     filetitle = 'AutoEncoder_{}_{}_{}_all_IO_noleave_{}.pt'.format(layer[0], layer[1], layer[2], set)
    # else:
    #     filetitle = 'AutoEncoder_{}_{}_{}_{}_all_IO_noleave_{}.pt'.format(layer[0], layer[1], layer[2], layer[3], set)

    # filetitle = 'AutoEncoder_{}_all_IO_noleave_{}.pt'.format(nodenum, set)

    # torch.save({
    #     'model_state_dict': model_state_dict_opt,
    #     # 'optimizer_state_dict': optimizer_state_dict_opt,
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
