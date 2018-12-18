def trainautoencoder_227(layer, set, result_draft_ratio):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import time
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from autoencoder_227 import autoencoder
    from sklearn.linear_model import LogisticRegression
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import copy

    batch_size = 8192
    learning_rate = 1e-3
    pickweight = 21.6
    outcomeweight = pickweight * result_draft_ratio

    datapath = '../../data/all/all_IO_noleave_' + set + '.npz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_print = 100

    model = autoencoder(layer).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print('Loading ' + datapath + ' on ' + device.type)
    dataset = np.load(datapath)

    TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
    ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
    TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())

    TrainX = np.hstack(
        (np.asarray(np.asmatrix(dataset['TrainX'])[0, 0].astype(np.float32).todense()), TrainY.astype(np.float32)))
    m_train = TrainX.shape[0]
    ValidX = np.hstack(
        (np.asarray(np.asmatrix(dataset['ValidX'])[0, 0].astype(np.float32).todense()), ValidY.astype(np.float32)))
    m_valid = ValidX.shape[0]
    TestX = np.hstack(
        (np.asarray(np.asmatrix(dataset['TestX'])[0, 0].astype(np.float32).todense()), TestY.astype(np.float32)))
    m_test = TestX.shape[0]

    n = TrainX.shape[1]

    TrainXae = torch.from_numpy(TrainX).to(device)
    dataloader_train = DataLoader(TrainXae, batch_size=batch_size, shuffle=False)

    ValidXae = torch.from_numpy(ValidX).to(device)
    dataloader_valid = DataLoader(ValidXae, batch_size=batch_size, shuffle=False)

    TestXae = torch.from_numpy(TestX).to(device)
    dataloader_test = DataLoader(TestXae, batch_size=batch_size, shuffle=False)

    if len(layer) == 2:
        nodenum = '{}-{}'.format(layer[0], layer[1])
    elif len(layer) == 3:
        nodenum = '{}-{}-{}'.format(layer[0], layer[1], layer[2])
    elif len(layer) == 4:
        nodenum = '{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3])
    elif len(layer) == 6:
        nodenum = '{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5])
    elif len(layer) == 10:
        nodenum = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5], layer[6], layer[7], layer[8], layer[9])

    print('Training Semi AutoEncoder of {}. Training set size:{}, batch size:{}'.format(nodenum, TrainXae.shape[0], batch_size))

    loss_opt = 2
    epoch_opt = 0
    model_state_dict_opt = model.state_dict()

    notoverfitting = True
    epochs = 0
    loss_train = np.zeros((0, 1))
    loss_valid = np.zeros((0, 1))
    acc_train = np.zeros((0, 1))
    acc_valid = np.zeros((0, 1))

    lr_raw = LogisticRegression(random_state=0, solver='newton-cg')
    lr_raw.fit(TrainX[:, :-1], TrainX[:, -1])
    score_train_lr_raw = lr_raw.score(TrainX[:, :-1], TrainX[:, -1])
    score_valid_lr_raw = lr_raw.score(ValidX[:, :-1], ValidX[:, -1])
    score_test_lr_raw = lr_raw.score(TestX[:, :-1], TestX[:, -1])

    print('Raw Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_raw,
                                                                                     100 * score_valid_lr_raw,
                                                                                     100 * score_test_lr_raw))

    while notoverfitting:
        start = time.time()
        i = 0
        loss_sum = 0
        acc_sum = 0
        for data in dataloader_train:
            i += 1
            _, recon = model(data[:, :-1])
            w = np.ones(data.shape).astype(np.float32)
            lw = np.where(data.cpu().data.numpy())
            w[lw[0], lw[1]] = pickweight
            w[:, -1] = outcomeweight
            criterion.weight = torch.from_numpy(w).to(device)
            loss = criterion(recon, data)
            loss_sum = loss_sum + loss.item()
            acc_sum = acc_sum + sum(data.cpu().data.numpy()[:, -1]==(recon.cpu().data.numpy()[:, -1] > 0.5).astype(int))
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        loss_train = np.vstack((loss_train, loss_sum / i))
        acc_train = np.vstack((acc_train, acc_sum / m_train))

        i = 0
        loss_sum = 0
        acc_sum = 0
        for data in dataloader_valid:
            i += 1
            _, recon = model(data[:, :-1])
            w = np.ones(data.shape).astype(np.float32)
            lw = np.where(data.cpu().data.numpy())
            w[lw[0], lw[1]] = pickweight
            w[:, -1] = outcomeweight
            criterion.weight = torch.from_numpy(w).to(device)
            loss_sum = loss_sum + criterion(recon, data).item()
            acc_sum = acc_sum + sum(
                data.cpu().data.numpy()[:, -1] == (recon.cpu().data.numpy()[:, -1] > 0.5).astype(int))
        loss_valid = np.vstack((loss_valid, loss_sum / i))
        acc_valid = np.vstack((acc_valid, acc_sum / m_valid))

        if loss_valid[-1, 0] < 0.99 * loss_opt:
            loss_opt = loss_valid[-1, 0]
            epoch_opt = epochs
            model_state_dict_opt = copy.deepcopy(model.state_dict())

        end = time.time()

        if epochs % n_print == 0:
            print(
                'Finished epoch {}, time {:.2f}s. Training loss:{:.4f}, Validation loss:{:.4f}'.format(
                    epochs, end - start, loss_train[-1, 0], loss_valid[-1, 0]))
            print('Comparing Reconstructed Draft with Actual Draft')
            recon = recon.cpu().data.numpy()
            l1 = np.where(data.cpu().data.numpy()[3, :-1])
            print('recon Empty/Hero Positions: {} / {}'.format(recon[3, l1[0] + 2], recon[3, l1]))

            # Validating prediction accuracy on the current best model, every print n iterations
            encode_train = np.zeros((0, layer[-1]))
            encode_valid = np.zeros((0, layer[-1]))
            encode_test = np.zeros((0, layer[-1]))
            recon_train = np.zeros((0, n))
            recon_valid = np.zeros((0, n))
            recon_test = np.zeros((0, n))

            model_opt = autoencoder(layer).to(device)
            model_opt.load_state_dict(model_state_dict_opt)

            i = 0
            loss_train_v = 0
            misrecon_train = 0
            acc_train_v = 0
            for data in dataloader_train:
                i += 1
                encode, recon = model_opt(data[:, :-1])
                w = np.ones(data.shape).astype(np.float32)
                lw = np.where(data.cpu().data.numpy())
                w[lw[0], lw[1]] = pickweight
                w[:, -1] = outcomeweight
                criterion.weight = torch.from_numpy(w).to(device)
                encode_train = np.vstack((encode_train, encode.cpu().data.numpy()))
                recon_train = np.vstack((recon_train, recon.cpu().data.numpy()))
                loss_train_v = loss_train_v + criterion(recon, data).item()
                recon_temp = recon.cpu().data.numpy()
                recon_round = (1 * (recon_temp[:, :-1] > 0.9).astype(int) + 0.5 * (
                            (recon_temp[:, :-1] >= 0.1) & (recon_temp[:, :-1] <= 0.9)).astype(int))
                misrecon_train = misrecon_train + sum(sum(data.cpu().data.numpy()[:, :-1] != recon_round))
                acc_train_v = acc_train_v + sum(data.cpu().data.numpy()[:, -1] == (recon_temp[:, -1] > 0.5).astype(int))
            loss_train_v = loss_train_v / i
            acc_train_v = acc_train_v / m_train

            i = 0
            loss_valid_v = 0
            misrecon_valid = 0
            acc_valid_v = 0
            for data in dataloader_valid:
                i += 1
                encode, recon = model_opt(data[:, :-1])
                w = np.ones(data.shape).astype(np.float32)
                lw = np.where(data.cpu().data.numpy())
                w[lw[0], lw[1]] = pickweight
                w[:, -1] = outcomeweight
                wt = torch.from_numpy(w).to(device)
                criterion.weight = wt
                encode_valid = np.vstack((encode_valid, encode.cpu().data.numpy()))
                recon_valid = np.vstack((recon_valid, recon.cpu().data.numpy()))
                loss_valid_v = loss_valid_v + criterion(recon, data).item()
                recon_temp = recon.cpu().data.numpy()
                recon_round = (1 * (recon_temp[:, :-1] > 0.9).astype(int) + 0.5 * (
                        (recon_temp[:, :-1] >= 0.1) & (recon_temp[:, :-1] <= 0.9)).astype(int))
                misrecon_valid = misrecon_valid + sum(sum(data.cpu().data.numpy()[:, :-1] != recon_round))
                acc_valid_v = acc_valid_v + sum(data.cpu().data.numpy()[:, -1] == (recon_temp[:, -1] > 0.5).astype(int))
            loss_valid_v = loss_valid_v / i
            acc_valid_v = acc_valid_v / m_valid

            i = 0
            loss_test = 0
            misrecon_test = 0
            acc_test = 0
            for data in dataloader_test:
                i += 1
                encode, recon = model_opt(data[:, :-1])
                w = np.ones(data.shape).astype(np.float32)
                lw = np.where(data.cpu().data.numpy())
                w[lw[0], lw[1]] = pickweight
                w[:, -1] = outcomeweight
                criterion.weight = torch.from_numpy(w).to(device)
                encode_test = np.vstack((encode_test, encode.cpu().data.numpy()))
                recon_test = np.vstack((recon_test, recon.cpu().data.numpy()))
                loss_test = loss_test + criterion(recon, data).item()
                recon_temp = recon.cpu().data.numpy()
                recon_round = (1 * (recon_temp[:, :-1] > 0.9).astype(int) + 0.5 * (
                        (recon_temp[:, :-1] >= 0.1) & (recon_temp[:, :-1] <= 0.9)).astype(int))
                misrecon_test = misrecon_test + sum(sum(data.cpu().data.numpy()[:, :-1] != recon_round))
                acc_test = acc_test + sum(data.cpu().data.numpy()[:, -1] == (recon_temp[:, -1] > 0.5).astype(int))
            loss_test = loss_test / i
            acc_test = acc_test / m_test

            print(
                'On {} dataset, Current weighted {} AutoEncoder T/V/T recon loss: {:.4f}/{:.4f}/{:.4f}, '
                'with recon mistake {}/{}/{}'
                    .format(set, nodenum, loss_train_v, loss_valid_v, loss_test, misrecon_train, misrecon_valid,
                            misrecon_test))
            print('Decoder Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * acc_train_v, 100 * acc_valid_v,
                                                                             100 * acc_test))

            lr = LogisticRegression(solver='newton-cg')
            lr.fit(encode_train, TrainY.ravel())
            score_train_lr = lr.score(encode_train, TrainY.ravel())
            score_valid_lr = lr.score(encode_valid, ValidY.ravel())
            score_test_lr = lr.score(encode_test, TestY.ravel())

            print('Encoding Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.
                  format(100 * score_train_lr, 100 * score_valid_lr, 100 * score_test_lr))

            lr_recon = LogisticRegression(random_state=0, solver='newton-cg')
            lr_recon.fit(recon_train[:, :-1], TrainY.ravel())
            score_train_lr_recon = lr_recon.score(recon_train[:, :-1], TrainY.ravel())
            score_valid_lr_recon = lr_recon.score(recon_valid[:, :-1], ValidY.ravel())
            score_test_lr_recon = lr_recon.score(recon_test[:, :-1], TestY.ravel())

            print('Recon LR Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_recon,
                                                                              100 * score_valid_lr_recon,
                                                                              100 * score_test_lr_recon))

        ############################
        # For test purpose only: short termination
        # if epochs > 2:
        #     notoverfitting = False
        ############################
        # run at least 100 epoch after the optimal model
        if epochs > 100 + epoch_opt:
            if loss_valid[-1, 0] > 1.1*loss_opt:
                notoverfitting = False
                print(
                'Training overfits at epoch {}; optimal model occurred at epoch {} with validation loss {:.4f}'.format(
                    epochs - 1, epoch_opt, loss_opt))
        elif epochs > 400:
            if epochs > 100 + epoch_opt:
                if (loss_valid[-1, 0] - loss_valid[-100, 0]) > -0.05*loss_valid[-1, 0]:
                    notoverfitting = False
                    print(
                    'Training stagnant at epoch {}; optimal model occurred at epoch {} with validation loss {:.4f}'.format(
                        epochs - 1, epoch_opt, loss_opt))
        elif epochs > 1500:
            print('Training exceeds 1500 epoch; optimal model at epoch {} with valid loss {:.4f}'.format(epoch_opt,
                                                                                                         loss_opt))
            notoverfitting = False

        epochs = epochs + 1

    ###### Commented out for testing purpose
    x = np.arange(0, epochs)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, loss_train, 'g', x, loss_valid, 'b')
    ax1.legend(('Training Loss', 'Validation Loss'))
    ax1.set_ylabel('Draft Reconstruction CE Loss')
    ax1.grid(True)
    ax2.plot(x, acc_train, 'g', x, acc_valid, 'b')
    ax2.legend(('Training Accuracy', 'Validation Accuracy'))
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Decoder Accuracy')
    ax2.grid(True)



    print('Encoding Train/Valid/Test sets...')

    encode_train = np.zeros((0, layer[-1]))
    encode_valid = np.zeros((0, layer[-1]))
    encode_test = np.zeros((0, layer[-1]))
    recon_train = np.zeros((0, n))
    recon_valid = np.zeros((0, n))
    recon_test = np.zeros((0, n))

    model_opt = autoencoder(layer).to(device)
    model_opt.load_state_dict(model_state_dict_opt)

    i = 0
    loss_train = 0
    misrecon_train = 0
    acc_train = 0
    for data in dataloader_train:
        i += 1
        encode, recon = model_opt(data[:, :-1])
        w = np.ones(data.shape).astype(np.float32)
        lw = np.where(data.cpu().data.numpy())
        w[lw[0], lw[1]] = pickweight
        w[:, -1] = outcomeweight
        criterion.weight = torch.from_numpy(w).to(device)
        encode_train = np.vstack((encode_train, encode.cpu().data.numpy()))
        recon_train = np.vstack((recon_train, recon.cpu().data.numpy()))
        loss_train = loss_train + criterion(recon, data).item()
        recon_temp = recon.cpu().data.numpy()
        recon_round = (1 * (recon_temp[:, :-1] > 0.9).astype(int) + 0.5 * (
                (recon_temp[:, :-1] >= 0.1) & (recon_temp[:, :-1] <= 0.9)).astype(int))
        misrecon_train = misrecon_train + sum(sum(data.cpu().data.numpy()[:, :-1] != recon_round))
        acc_train = acc_train + sum(data.cpu().data.numpy()[:, -1] == (recon_temp[:, -1] > 0.5).astype(int))
    loss_train = loss_train / i
    acc_train = acc_train / m_train

    i = 0
    loss_valid = 0
    misrecon_valid = 0
    acc_valid = 0
    for data in dataloader_valid:
        i += 1
        encode, recon = model_opt(data[:, :-1])
        w = np.ones(data.shape).astype(np.float32)
        lw = np.where(data.cpu().data.numpy())
        w[lw[0], lw[1]] = pickweight
        w[:, -1] = outcomeweight
        wt = torch.from_numpy(w).to(device)
        criterion.weight = wt
        encode_valid = np.vstack((encode_valid, encode.cpu().data.numpy()))
        recon_valid = np.vstack((recon_valid, recon.cpu().data.numpy()))
        loss_valid = loss_valid + criterion(recon, data).item()
        recon_temp = recon.cpu().data.numpy()
        recon_round = (1 * (recon_temp[:, :-1] > 0.9).astype(int) + 0.5 * (
                (recon_temp[:, :-1] >= 0.1) & (recon_temp[:, :-1] <= 0.9)).astype(int))
        misrecon_valid = misrecon_valid + sum(sum(data.cpu().data.numpy()[:, :-1] != recon_round))
        acc_valid = acc_valid + sum(data.cpu().data.numpy()[:, -1] == (recon_temp[:, -1] > 0.5).astype(int))
    loss_valid = loss_valid / i
    acc_valid = acc_valid / m_valid

    i = 0
    loss_test = 0
    misrecon_test = 0
    acc_test = 0
    for data in dataloader_test:
        i += 1
        encode, recon = model_opt(data[:, :-1])
        w = np.ones(data.shape).astype(np.float32)
        lw = np.where(data.cpu().data.numpy())
        w[lw[0], lw[1]] = pickweight
        w[:, -1] = outcomeweight
        criterion.weight = torch.from_numpy(w).to(device)
        encode_test = np.vstack((encode_test, encode.cpu().data.numpy()))
        recon_test = np.vstack((recon_test, recon.cpu().data.numpy()))
        loss_test = loss_test + criterion(recon, data).item()
        recon_temp = recon.cpu().data.numpy()
        recon_round = (1 * (recon_temp[:, :-1] > 0.9).astype(int) + 0.5 * (
                (recon_temp[:, :-1] >= 0.1) & (recon_temp[:, :-1] <= 0.9)).astype(int))
        misrecon_test = misrecon_test + sum(sum(data.cpu().data.numpy()[:, :-1] != recon_round))
        acc_test = acc_test + sum(data.cpu().data.numpy()[:, -1] == (recon_temp[:, -1] > 0.5).astype(int))
    loss_test = loss_test / i
    acc_test = acc_test / m_test

    print('On {} dataset, optimal {} AutoEncoder T/V/T loss: {:.4f}/{:.4f}/{:.4f}'.format(set, nodenum, loss_train,
                                                                                          loss_valid, loss_test))

    print('Comparing Reconstructed Draft with Actual Draft')
    g1=33

    l1 = np.where(TrainX[g1, :-1])
    print('Game {}, heros {}'.format(g1 + 1, l1[0]))
    print('Comparing Reconstructed Draft with Actual Draft')
    print('Actual slots: {}/{}'.format(TrainX[g1, l1[0] + 2], TrainX[g1, l1]))
    print('Recon slots: {}/{}'.format(recon_train[g1, l1[0] + 2], recon_train[g1, l1]))

    print('Decoder Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(acc_train * 100, acc_valid * 100, acc_test * 100))

    lr_encode = LogisticRegression(random_state=0, solver='newton-cg')
    lr_encode.fit(encode_train, TrainY.ravel())
    score_train_lr_encode = lr_encode.score(encode_train, TrainY.ravel())
    score_valid_lr_encode = lr_encode.score(encode_valid, ValidY.ravel())
    score_test_lr_encode = lr_encode.score(encode_test, TestY.ravel())

    print('Encoding Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_encode,
                                                                                 100 * score_valid_lr_encode,
                                                                                 100 * score_test_lr_encode))

    lr_recon = LogisticRegression(random_state=0, solver='newton-cg')
    lr_recon.fit(recon_train[:, :-1], TrainY.ravel())
    score_train_lr_recon = lr_recon.score(recon_train[:, :-1], TrainY.ravel())
    score_valid_lr_recon = lr_recon.score(recon_valid[:, :-1], ValidY.ravel())
    score_test_lr_recon = lr_recon.score(recon_test[:, :-1], TestY.ravel())

    print('Recon LR Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_recon,
                                                                      100 * score_valid_lr_recon,
                                                                      100 * score_test_lr_recon))

    print('Raw Logistic Regression Prediction Accuracy: {:.2f}/{:.2f}/{:.2f}'.format(100 * score_train_lr_raw,
                                                                                     100 * score_valid_lr_raw,
                                                                                     100 * score_test_lr_raw))

    print('On {} dataset, optimal {} semi-AE T/V/T loss: {:.4f}/{:.4f}/{:.4f}'.format(set, nodenum, loss_train,
                                                                                          loss_valid, loss_test))

    ax1.set_title(
        'Semi-AE of layer {} on {} data, Result-to-Recon ratio {:.1f}'
        '\n valid recon loss {:.4f}, encode LR accuracy {:2.2f}%'.format(nodenum, set, outcomeweight / pickweight,
                                                                         loss_opt, 100 * score_valid_lr_encode))

    ax2.set_title('Decoder Prediction Accuracy {:2.2f}% on Test Set'.format(acc_test * 100))

    fig.savefig('Semi_AutoEncoder_{}_all_IO_noleave_{}.jpg'.format(nodenum, set))

    filetitle = 'Semi_AutoEncoder_{}_all_IO_noleave_{}.pt'.format(nodenum, set)

    torch.save({
        'model_state_dict': model_state_dict_opt,
        'layer': layer,
        'epochs': epochs,
        'epoch_optimal': epoch_opt,
        'loss_optimal': loss_opt,
        'model_loss': [loss_train, loss_valid, loss_test],
        'decoder_accuracy': [acc_train, acc_valid, acc_test],
        'lr_score': [score_train_lr_encode, score_valid_lr_encode, score_test_lr_encode]
    }, filetitle)
    # return acc_valid, score_valid_lr_encode
