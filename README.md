## Pattern-based Neural Language Model
Code for the small Pat-Sum model from the paper [Patterns versus Characters in Subword-aware Neural Language Models](https://arxiv.org/abs/1709.00541) (ICONIP 2017)

### Requirements
Code is written in Python 3 and requires TensorFlow 1.3+. It also requires the following Python modules: `numpy`,  `argparse`. You can install them via:
```
sudo pip3 install numpy argparse
```

### Data
Raw data should be put into the `data_c16/` directory, split into `train.txt`, `valid.txt`, and `test.txt`. Each line of the .txt file should be a sentence. Spaces (` `) need to be replaced by underscores (`_`). The WikiText-2 (WT2) data is given as the default.

Converted (chars -> patterns) data should be put into the `data_c16/` directory, split into `trainFSM.txt`, `validFSM.txt`, and `testFSM.txt`. Also, these three files should be concatenated in `allFSM.txt`. Each line of the .txt file should be a sentence. The WikiText-2 (WT2) data is given as the default.

The code for pattern-mining and conversion can be requested from [Rustem Takhanov](https://sst.nu.edu.kz/rustem-takhanov/).

### Model
To reproduce the result for small Pat-Sum from Table 2
```
python3 VD-LSTM-Pat-Sum.py
```

### Other options
To see the full list of options run
```
python3 VD-LSTM-Pat-Sum.py -h
```
