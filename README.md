# CSC2515 Dota2 Draft Prediction Project
2018-11-02: Note that Semenov's dataset is too large to be uploaded to Github, not to mention the huge binary matrices

### Report   
Report is available [here](https://github.com/risingdhxs/CSC2515Dota2DraftPredictionProject/blob/master/report/csc2515.pdf).


### Results by Shawn


I've tried 12 different fully connected NN and provided the results accordingly here. All of the results could be found in [CSC2515Dota2DraftPredictionProject/Algorithms/shawn/accuracy/](https://github.com/risingdhxs/CSC2515Dota2DraftPredictionProject/tree/master/Algorithms/shawn/accuracy).

- FCs_summary.xlsx: I've included the structures of the NN, epochs, accuracies and loss into this file.

- exp: I ran each of 12 NNs over normal data and place the results into this folder. There are three types of files
    - \*.batch.pth: the best model saved when the best validation accuracy is achieved.
    - \*.batch.log: the log of each experiments
    - \*.batch.log.png: the corresponding accuracy graphs
    - \*.batch.log.loss.png: the corresponding loss graphs
    Besides, for each fc_n.*, n means the nth NN.

- exp_VH: I ran each of 12 NNs over VH data and place the results into this folder.

### Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

#### Changed [0.0.1] - 2018-11-18
##### File Structure
- **Data** folder contains all the Dota2 matches.
    - All previous used data are now moved to **data/obsolete**. 
    - **data/all**: the IO data of all matches including training set, validation set and test set.
    - **data/test**: test set. (All, Normal level, Very high level)
    - **data/training**: training set. (All, Normal level, Very high level)
    - **data/validation**: validation set. (All, Normal level, Very high level)
    - **data/SVD**.
    - Semenov_small_data_preview.csv
- **Algorithms** folder contains all the implemented models.
    - **algorithms/benchmark**: including Logistic Regression, SVM and Naive Bayes.
    - **algorithms/data_processing**: codes for file processing.
    - **algorithms/saved_models**: trained machine learning models saved in pickle object file.
