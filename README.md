# kth-signal-peptide-prediction
This is a repository for project at KTH relate to subject DD2404
Description:

## Repository structure
/bin . All python code are stored under /bin folder,

/data . All data are stored under /data folder

/results . All Results are stored under /results folder, with each test-run stored into corresponding date

/doc . All documentations relate to this project are store under /doc

/doc/log_book stored logs relate to this project

## How to use the code to train and predict signal peptides?
Under /bin folder, run ./runall SAVEPATH [argumemts]
```
usage: runall [-h] [-pt] [--nfold NFOLD] [-f] [-p]
                            [-t THRESHOLD] [-i INPUT_MODE] [-q]
                            save_path
```

There are 3 function groups, they are 1.Performance Test, 2.F-score test, 3. Prediction

1. Perfromance test (-pt) train the model based on the data specified by [-i INPUT_MODE], and produce a graph thay compares all models. The graph is saved in the path speicifed by SAVEPATH. 
   --nfold can be used as an additional argument to spcify the number of fold for cross-validation. Default to 3.
   ```
   example: ./runall ../results/2017-12-26 -pt --nfold 5
   ```
   Above command will run performance test with nfold of 5 and save to the results folder

2. F-score test (-f) train the models with data specified by INPUT_MODE based on 80/20 data split, and save precision/recall and f-beta score into SAVEPATH
   ```
   example ./runall ../results/2017-12-26 -f -i 1
   ```
   Above code will run f-score test with non_TM data only
   
3. Prediction (-p), Train Logistic regression and run predictions on files located under /data/proteomes/ ,files must be in fasta format.
    -t can be used to indicate the THRESHOLD for classify a sample into positive. Default to 0.5

[-i INPUT_MODE] is the argument for specify which data is used for training. 
  * 1 = non-TM only, 
  * 2 = TM only, 
  * All other numbers = TM + non-TM will be usesd
  
 [-q] quiet mode, will not print anything on screen.

