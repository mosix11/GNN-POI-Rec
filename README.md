# Next POI Recommendation Using Spatio-Temporal GNN

This repository contains the implementation of the Hierarchical Multitask Graph Recurrent Network proposed in the paper [Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation](https://dl.acm.org/doi/pdf/10.1145/3477495.3531989). The impelementaion slightly differs from the original method since I applied it to the smaller [Foursquare NYC](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46) dataset. Also, this implementation uses operations that makes training a lot faster. The model significantly outperforms the baseline model which is a simple LSTM.

## Installation

In addition to the requirements listed in `requirements.txt`, the project requires `p7zip` for dataset extraction.

1. Clone the repository:

    ```bash
    git clone https://github.com/mosix11/GNN-POI-Rec.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Structure

```plaintext
📂 .
├── 📂 configs/                 
│   ├── 📄 baseline.yaml        # Config for baseline model
│   └── 📄 hmt_grn.yaml         # Config for HMT-GRN model
├── 📂 src/                     
│   ├── 📂 dataset/             
│   │   ├── 📄 FoursquareNYC.py # Dataset loader for Foursquare NYC
│   │   └── 📄 trajectory_dataset.py 
│   ├── 📂 metrics/             
│   │   ├── 📄 AccuracyK.py     # Script for Accuracy@K metric
│   │   └── 📄 MRR.py           # Script for Mean Reciprocal Rank (MRR) metric
│   ├── 📂 models/              
│   │   ├── 📄 baseline.py      # Baseline model
│   │   ├── 📄 HMT_GRN_V2.py    # Main model
│   ├── 📂 trainer/             
│   │   └── 📄 trainer.py       
│   └── 📂 utils/               
├── 📄 hyp_tuning.py            # Script for hyperparameter tuning
└── 📄 train.py                 # Main training script
```

## Usage

The training configuration is specified in the `configs/hmt_grn.yaml` file. You can change it if you want to modify the model or training hyperparameters.
To start the training, you simply run the train script:

```bash
python train.py -m grn -c hmt_grn.yaml
```

The evaluaation will be done right after the training is finished.

In order to run hyperparameter search you can run:

```bash
python hyp_tuning.py
```

## Performance

The following table shows the performance of the model with the last 6 check-ins of each user being kept for test split.

```plaintext
╭──────────┬──────────╮
│ Metric   │    Score │
├──────────┼──────────┤
│ Acc@1    │ 0.239583 │
├──────────┼──────────┤
│ Acc@5    │ 0.487132 │
├──────────┼──────────┤
│ Acc@10   │ 0.534467 │
├──────────┼──────────┤
│ Acc@20   │ 0.550398 │
├──────────┼──────────┤
│ MRR      │ 0.36405  │
╰──────────┴──────────╯
```

## References

```bibtex
@inproceedings{10.1145/3477495.3531989,
  author = {Mehrotra, Rishabh and Anderson, Ashton and Pera, Maria Soledad and Shah, Chirag},
  title = {RecSys ’22: The 16th ACM Conference on Recommender Systems},
  booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
  year = {2022},
  isbn = {9781450392743},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3477495.3531989},
  url = {https://doi.org/10.1145/3477495.3531989},
}
```
