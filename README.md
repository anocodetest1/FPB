# AFPB

To create exceutable environment with anaconda:
```
conda env create -f conda.yaml
conda activate reid
```

To install required python packages with pip:
```
pip install -r requirements.txt
```

To train and evaluate AFPB model on Market1501 dataset:
```
python exp_AFPB.py --config-file ./configs/market.yaml --root /path/to/your/data
```

To train and evaluate AFPB model on CUHK03_Detected dataset:
```
python exp_AFPB.py --config-file ./configs/cuhk_detected.yaml --root /path/to/your/data
```

To train and evaluate AFPB model on CUHK03_Labeled dataset:
```
python exp_AFPB.py --config-file ./configs/cuhk_labeled.yaml --root /path/to/your/data
```

To train and evaluate AFPB model on DukeMTMC dataset:
```
python exp_AFPB.py --config-file ./configs/duke.yaml --root /path/to/your/data
```

To train and evaluate AFPB model on msmt17 dataset:
```
python exp_AFPB.py --config-file ./configs/msmt17.yaml --root /path/to/your/data
```

All records and log file are in the `./log/` directory.