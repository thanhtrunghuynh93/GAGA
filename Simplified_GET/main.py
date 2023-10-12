import pandas as pd

from model import GET_Model
from data_utils import QueryDataset, Vocabulary, QueryTorchDataset
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pickle

def build_vocabs(train_data):

    vocab = Vocabulary()
    vocab.fit(train_data.get_text_corpus())
    evd_vocab = Vocabulary()
    evd_vocab.fit(train_data.get_src_corpus())

    return vocab, evd_vocab


def build_word_embedding_matrix(
    vocab, glove_path="/home/hoangdzung/Downloads/glove.6B/glove.6B.100d.txt"
):
    term_index = vocab._state["term_index"]
    embedding_data = {}
    output_dim = 0
    count_word_hit = 0

    with open(glove_path, "r", encoding="utf-8") as f:
        output_dim = len(f.readline().rstrip().split(" ")) - 1
        f.seek(0)
        for line in tqdm(f):
            current_line = line.rstrip().split(" ")
            if current_line[0] not in term_index:
                continue
            embedding_data[current_line[0]] = current_line[1:]
            count_word_hit += 1

        print(
            "Word hit: "
            + str((count_word_hit, len(term_index)))
            + " "
            + str(count_word_hit / len(term_index) * 100)
        )

    input_dim = len(term_index)
    matrix = np.empty((input_dim, output_dim))
    valid_keys = embedding_data.keys()
    # Starting the smallest index to the largest
    for term, index in sorted(term_index.items(), key=lambda x: x[1]):
        if term in valid_keys:
            matrix[index] = embedding_data[term]
        else:
            matrix[index] = np.random.uniform(-0.2, 0.2)
    return matrix


def collate_batch(batch):
    (
        query_list,
        query_adj_list,
        query_mask_list,
        evd_list,
        evd_adj_list,
        evd_mask_list,
        n_evd_list,
        label_list,
        orig_left_id_list,
    ) = ([], [], [], [], [], [], [], [], [])

    for query, query_adj, query_mask, evds, evd_adjs, evd_masks, n_evds, label, orig_left_id in batch:
        query_list.append(query)
        query_adj_list.append(query_adj)
        query_mask_list.append(query_mask)
        evd_list.append(evds)
        evd_adj_list.append(evd_adjs)
        evd_mask_list.append(evd_masks)
        n_evd_list.append(n_evds)
        label_list.append(label)
        orig_left_id_list.append(orig_left_id)

    query_list = torch.LongTensor(np.stack(query_list))
    query_adj_list = torch.FloatTensor(np.stack(query_adj_list))
    query_mask_list = torch.LongTensor(np.stack(query_mask_list))

    evd_list = torch.LongTensor(np.vstack(evd_list))
    evd_adj_list = torch.FloatTensor(np.vstack(evd_adj_list))
    evd_mask_list = torch.LongTensor(np.vstack(evd_mask_list))

    n_evd_list = torch.LongTensor(n_evd_list)
    label_list = torch.LongTensor(label_list)
    orig_left_id_list = torch.LongTensor(orig_left_id_list)
    return (
        query_list,
        query_adj_list,
        query_mask_list,
        evd_list,
        evd_adj_list,
        evd_mask_list,
        n_evd_list,
    ), label_list, orig_left_id_list


def main(args):
    df_train = pd.read_csv(args.train_file, delimiter="\t")
    df_test = pd.read_csv(args.test_file, delimiter="\t")
    df_dev = pd.read_csv(args.dev_file, delimiter="\t")

    for df in [df_train, df_test, df_dev]:
        print(
            "There are at most {} evidences per claim".format(
                df.groupby("id_left").count().cred_label.max()
            )
        )

    train_data = QueryDataset(df_train)
    dev_data = QueryDataset(df_dev)
    test_data = QueryDataset(df_test)

    vocab, evd_vocab = build_vocabs(train_data)

    train_data.encoding(vocab, evd_vocab)
    dev_data.encoding(vocab, evd_vocab)
    test_data.encoding(vocab, evd_vocab)

    train_data.build_data()
    dev_data.build_data()
    test_data.build_data()

    train_dataset = QueryTorchDataset(train_data)
    dev_dataset = QueryTorchDataset(dev_data)
    test_dataset = QueryTorchDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, collate_fn=collate_batch, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        shuffle=False,
    )

    word_embedding_matrix = build_word_embedding_matrix(vocab, args.glove_path)

    get_model = GET_Model(
        input_dim=word_embedding_matrix.shape[1],
        hidden_dim=args.hidden_dim,
        word_embedding=torch.FloatTensor(word_embedding_matrix),
        max_num_evd=30,
        word_att_num_heads=5,
        evd_att_num_heads=2,
    ).cuda()

    optimizer = torch.optim.Adam(
        get_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0
        for x, y, _ in tqdm(train_loader):
            x = tuple([i.cuda() for i in x])
            y = y.cuda()
            _, logits = get_model(*x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(get_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            # break
        print(epoch, total_loss)

    get_model.eval()

    with torch.no_grad():
        embeddings = {}
        for loader in [train_loader, dev_loader, test_loader]:
            all_logits = []
            all_trues = []
            for x, y, ori_ids in tqdm(loader):
                x = tuple([i.cuda() for i in x])
                y = y.cuda()
                embs, logits = get_model(*x)
                all_logits.append(logits)
                all_trues.append(y)

                embs = embs.cpu().numpy()
                ori_ids = ori_ids.cpu().numpy()
                for ori_id, emb in zip(ori_ids, embs):
                    embeddings[ori_id] = emb 

            all_trues = torch.cat(all_trues).detach().cpu().numpy()
            all_logits = torch.cat(all_logits, dim=0)
            predicts = all_logits.argmax(dim=1).detach().cpu().numpy()

            print("F1_micro:", f1_score(all_trues, predicts, average="micro"))
            print("F1_macro:", f1_score(all_trues, predicts, average="macro"))

    if args.outpath:
        assert len(embeddings) == len(train_dataset) + len(dev_dataset) + len(test_dataset)
        pickle.dump(embeddings, open(args.outpath, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        help="TSV train file in GET folder",
        default="/home/hoangdzung/Documents/EPFL/fakenews/GET//formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="TSV test file in GET folder",
        default="/home/hoangdzung/Documents/EPFL/fakenews/GET//formatted_data/declare/Snopes/mapped_data/5fold/test_0.tsv",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        help="TSV dev file in GET folder",
        default="/home/hoangdzung/Documents/EPFL/fakenews/GET//formatted_data/declare/Snopes/mapped_data/dev.tsv",
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        help="Path to glove embedding file",
        default="/home/hoangdzung/Downloads/glove.6B/glove.6B.100d.txt",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=300, help="Hidden dim, default=300"
    )
    parser.add_argument(
        "--max_num_evd,",
        type=int,
        default=30,
        help="The maximum number of evidence to be considered per query, default=30",
    )
    parser.add_argument(
        "--word_att_num_heads,",
        type=int,
        default=5,
        help="Number of attention heads in the word level, default=5",
    )
    parser.add_argument(
        "--evd_att_num_heads,",
        type=int,
        default=2,
        help="Number of attention heads in the evidence level, default=2",
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate, default = 1e-4"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay, default = 1e-3"
    )
    parser.add_argument(
        "--outpath",
        type=str,
        help="Path of numpy output file which saves the embeddings",
    )
    args = parser.parse_args()
    main(args)
