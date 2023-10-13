<h1> Code repository for Homophily fake news detection framework </h1>

The code is the combination of reproduced code from the two papers: 

GET: Evidence-aware Fake News Detection with Graph Neural Networks (<a href = "https://arxiv.org/pdf/2201.06885.pdf"> link </a>)

GAGA: Label Information Enhanced Fraud Detection against Low Homophily in Graphs (<a href = "https://arxiv.org/pdf/2302.10407.pdf"> link </a>)

Unzip the data and preprocessed data, the run file are in the folders

# How to run
## Get embeddings fron GET
1. Go to https://nlp.stanford.edu/projects/glove/ to download glove embedding file and put it into `data/glove/embedding`
2. Run `Simplified_GET/main.py` to get the embedding of each claim. We save them as a dictionary in pickle format.
``` bash
python3 Simplified_GET/main.py \
--train_file data/formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv \
--test_file data/formatted_data/declare/Snopes/mapped_data/5fold/test_0.tsv \
--dev_file data/formatted_data/declare/Snopes/mapped_data/dev.tsv \
--glove_path data/glove_embedding/glove.6B.100d.txt \
--batch_size 8 \
--epochs 10 \
--outpath data/embeddings/embeddings_Snopes_0.pkl
```

## Build the graphs
1. Run `graph4GAGA.py` to build the dgl graphs and save them into binary files. The `train_file`, `test_file`, `dev_file` must be the same as the first step, while the `embedding_file` must be the same as `outpath` in the first step. For example
``` bash
python3 graph4GAGA.py 
--train_file data/formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv \
--test_file data/formatted_data/declare/Snopes/mapped_data/5fold/test_0.tsv \
--dev_file data/formatted_data/declare/Snopes/mapped_data/dev.tsv \
--embedding_file data/embeddings/embeddings_Snopes_0.pkl \
--outpath data/graphs/Snopes_0.bin