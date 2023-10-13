dataset="Snopes"
fold_id=0
graph_id=0
train_file="data/formatted_data/declare/${dataset}/mapped_data/5fold/train_${fold_id}.tsv "
test_file="data/formatted_data/declare/${dataset}/mapped_data/5fold/test_${fold_id}.tsv "
dev_file="data/formatted_data/declare/${dataset}/mapped_data/dev.tsv "
embedding_file="data/embeddings/embeddings_${dataset}_${fold_id}.pkl"
graph_file="data/graphs/${dataset}_${fold_id}.pkl"
config_gaga="GAGA/pytorch_gaga/configs/${dataset}_${fold_id}_${graph_id}.json"

if [ ! -f ${config_gaga} ]
then
    echo "Config file does not exist in Bash. Please create a config file based on GAGA/pytorch_gaga/configs/Snopes_0_0.json"
else
    echo "Config file found. Please double check that graph_id and dataset name in the config file are correct"
fi

python3 Simplified_GET/main.py \
--train_file ${train_file} \
--test_file ${test_file} \
--dev_file ${dev_file} \
--glove_path data/glove_embedding/glove.6B.100d.txt \
--batch_size 8 \
--epochs 1 \
--outpath ${embedding_file}

python3 graph4GAGA.py \
--train_file ${train_file} \
--test_file ${test_file} \
--dev_file ${dev_file} \
--embedding_file ${embedding_file} \
--outpath "data/graphs/${dataset}_${fold_id}.bin"

python3 GAGA/pytorch_gaga/preprocessing/dataset_split.py \
--dataset "data/graphs/${dataset}_${fold_id}" \
--save_dir data/gaga_sequence_data/ \
--graph_id ${graph_id}

python3 GAGA/pytorch_gaga/preprocessing/graph2seq_mp.py \
--dataset "data/graphs/${dataset}_${fold_id}" \
--fanouts -1 -1 --save_dir data/gaga_sequence_data \
--add_self_loop --norm_feat --graph_id ${graph_id}

python3 GAGA/pytorch_gaga/main_transformer.py \
--config ${config_gaga} \
--gpu 0  --log_dir logs --early_stop 100 --seeds 1 2 3