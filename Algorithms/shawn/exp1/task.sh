# using the original dataset loaded from data/all/all_IO_noleave_N.npz
python fc_with_minibatch.py 1 200 ./ 10000 &&\
python fc_with_minibatch.py 2 200 ./ 10000 &&\
python fc_with_minibatch.py 3 400 ./ 10000 &&\
python fc_with_minibatch.py 4 400 ./ 10000 &&\
python fc_with_minibatch.py 5 600 ./ 10000 &&\
python fc_with_minibatch.py 6 600 ./ 10000
