python fc_with_minibatch.py 1 200 ./exp 10000 1  &&\
python fc_with_minibatch.py 2 200 ./exp 10000 1  &&\
python fc_with_minibatch.py 3 600 ./exp 10000 1  &&\
python fc_with_minibatch.py 4 600 ./exp 10000 1  &&\
python fc_with_minibatch.py 5 800 ./exp 10000 1  &&\
python fc_with_minibatch.py 6 800 ./exp 10000 1  &&\
python fc_with_minibatch.py 7 200 ./exp 10000 1  &&\
python fc_with_minibatch.py 8 200 ./exp 10000 1  &&\
python fc_with_minibatch.py 9 600 ./exp 10000 1  &&\
python fc_with_minibatch.py 10 600 ./exp 10000 1  &&\
python fc_with_minibatch.py 11 800 ./exp 10000 1  &&\
python fc_with_minibatch.py 12 800 ./exp 10000 1  &&\
python fc_with_minibatch.py 1 200 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 2 200 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 3 600 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 4 600 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 5 800 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 6 800 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 7 200 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 8 200 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 9 600 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 10 600 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 11 800 ./exp_VH 10000 2  &&\
python fc_with_minibatch.py 12 800 ./exp_VH 10000 2



% python draw.py fc_1.batch.log fc_1.batch.log single_layer_with_no_dropout
% python draw.py fc_2.batch.log fc_2.batch.log single_layer_with_dropout
% python draw.py fc_3.batch.log fc_3.batch.log double_layer_with_no_dropout
% python draw.py fc_4.batch.log fc_4.batch.log double_layer_with_dropout
% python draw.py fc_5.batch.log fc_5.batch.log triple_layer_with_no_dropout
% python draw.py fc_6.batch.log fc_6.batch.log triple_layer_with_dropout
% 
% python draw.py fc_7.batch.log fc_7.batch.log single_layer_with_no_dropout
% python draw.py fc_8.batch.log fc_8.batch.log single_layer_with_dropout
% python draw.py fc_9.batch.log fc_9.batch.log double_layer_with_no_dropout
% python draw.py fc_10.batch.log fc_10.batch.log double_layer_with_dropout
% python draw.py fc_11.batch.log fc_11.batch.log triple_layer_with_no_dropout
% python draw.py fc_12.batch.log fc_12.batch.log triple_layer_with_dropout
