<h1> Code repository for Homophily fake news detection framework </h1>

The code is the combination of reproduced code from the two papers: 

GET: Evidence-aware Fake News Detection with Graph Neural Networks (<a href = "https://arxiv.org/pdf/2201.06885.pdf"> link </a>)

GAGA: Label Information Enhanced Fraud Detection against Low Homophily in Graphs (<a href = "https://arxiv.org/pdf/2302.10407.pdf"> link </a>)

# How to run
## Step by step
### Get embeddings from GET
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

### Build the graphs
1. Run `graph4GAGA.py` to build the dgl graphs and save them into binary files. The `--train_file`, `--test_file`, `--dev_file` must be the same as the first step, while the `--embedding_file` must be the same as `--outpath` in the first step. For example
``` bash
python3 graph4GAGA.py \
--train_file data/formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv \
--test_file data/formatted_data/declare/Snopes/mapped_data/5fold/test_0.tsv \
--dev_file data/formatted_data/declare/Snopes/mapped_data/dev.tsv \
--embedding_file data/embeddings/embeddings_Snopes_0.pkl \
--outpath data/graphs/Snopes_0.bin
```
We consider 6 directed edge types:
- `mutual_evd`: Two claims have a mutual evidence source.
- `1-1`: Two claims are both supported by an evidence source.
- `1-0`: src vertex claim is supported by an evidence source while that evidence source doesn't support the dst vertex claim
- `0-1`: src vertex claim isn't supported by an evidence source while that evidence source does support the dst vertex claim
- `0-0`: Two claims are not supported by an evidence source
- `np`: Two claims have at least one common noun phrase

Then 4 graphs are created with different subsets of edge types:
- `1-1`, `0-0`, `1-0`,`0-1`, `np`
- `1-1`, `0-0`, `1-0`,`0-1`, `np`
- `mutual_evd`, `np`
- `mutual_evd`

### Run GAGA
1. Create the graph sequence from the binary graph file by `GAGA/pytorch_gaga/preprocessing/graph2seq_mp.py`
```bash
python3 GAGA/pytorch_gaga/preprocessing/dataset_split.py --dataset data/graphs/Snopes_0 --save_dir data/gaga_sequence_data/ --graph_id 0

python3 GAGA/pytorch_gaga/preprocessing/graph2seq_mp.py --dataset data/graphs/Snopes_0 --fanouts -1 -1 --save_dir data/gaga_sequence_data --add_self_loop --norm_feat --graph_id 0
```
Note that we don't use `--train_size`, `--val_size` since we use the benchmark train test split. These args are only used for randomized train test split.
`--graph_id` can be from 0 to 3, depending on which edge types you want to use (check the previous step).

2. Run GAGA
```bash
python3 GAGA/pytorch_gaga/main_transformer.py --config GAGA/pytorch_gaga/configs/Snopes_0_0.json --gpu 0  --log_dir logs --early_stop 100 --seeds 1 2 3
```

## All at once
You can modify and run `run.sh` to do all the above steps. 

# TODO:
Add creating the sentiment edges (`1-1`, `0-0`, `1-0`, `0-1`) as an option since running the NLI model is heavy and takes time.
