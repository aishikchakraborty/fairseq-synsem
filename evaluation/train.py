import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import argparse
import csv
import os
import json
from tqdm import tqdm
import time
import mmap
import _pickle as pickle

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/multinli_1.0/',
                    help='location of the data corpus')
parser.add_argument('--emb', type=str, default='../embeddings',
                    help='location of the data corpus')
parser.add_argument('--random_seed', type=int, default=13370,
                    help='random seed')
parser.add_argument('--numpy_seed', type=int, default=1337,
                    help='numpy random seed')
parser.add_argument('--torch_seed', type=int, default=133,
                    help='pytorch random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int, default=0,
                    help='use gpu x')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='epochs')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--save', type=str, default='models/',
                    help='path to save the final model')
parser.add_argument('--hidden-dim', type=int, default=500,
                    help='hidden dimension size')
parser.add_argument('--save-emb', type=str, default='embeddings/',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

args = parser.parse_args()

print(args)
if args.random_seed is not None:
    random.seed(args.random_seed)
if args.numpy_seed is not None:
    np.random.seed(args.numpy_seed)
if args.torch_seed is not None:
    torch.manual_seed(args.torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

print('Load vocab and embeddings...')
vocab = torch.load(args.emb + '/vocab.pb')
print(vocab[:10])
w2idx = {w:idx for idx, w in enumerate(vocab)}

emb = torch.load(args.emb + '/emb_matrix.pb')
sem_emb = torch.load(args.emb + '/sem_matrix.pb')
syn_emb = torch.load(args.emb + '/en_syn_matrix.pb')

embedding_dim = emb.size(1)
print(len(vocab))

print('Loaded vocab and embeddings')

gold_label2idx = {'entailment': 0, 'contradiction': 1,'neutral': 2}

def convert_seq_to_id(sent):
    words_list = sent.split()
    ids_list = []
    for w in words_list:
        if w in vocab:
            ids_list.append(w2idx[w])
        else:
            ids_list.append(w2idx['<unk>'])
    return ids_list


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_dataset(mode='train'):
    s1, s2, label = [], [], []
    with open(os.path.join(args.data, 'multinli_1.0_' + mode + '.jsonl'), 'r') as f:
        for lines in tqdm(f, total=get_num_lines(os.path.join(args.data, 'multinli_1.0_' + mode + '.jsonl'))):
            lines = json.loads(lines.rstrip('\n'))
            try:
                label.append(gold_label2idx[lines['gold_label']])
                s1.append(convert_seq_to_id(lines['sentence1']))
                s2.append(convert_seq_to_id(lines['sentence2']))
            except:
                continue

    return s1, s2, label

print('Loading dataset...')

if os.path.exists('data/multinli_train.pb') and os.path.exists('data/multinli_val.pb'):
    train_data = pickle.load(open('data/multinli_train.pb', 'rb'))
    val_data = pickle.load(open('data/multinli_val.pb', 'rb'))
    # test_data = pickle.load(open('data/multinli_test.pb', 'rb'))

    train_s1 , train_s2, train_y = train_data[0], train_data[1], train_data[2]
    val_s1 , val_s2, val_y = val_data[0], val_data[1], val_data[2]

    assert len(train_s1) == len(train_y), "Length Mismatch"
    assert len(train_s2) == len(train_y), "Length Mismatch"

    assert len(val_s1) == len(val_y), "Length Mismatch"
    assert len(val_s1) == len(val_y), "Length Mismatch"


    # test_s1 , test_s2, test_y = test_data[0], test_data[1], test_data[2]
else:
    train_s1 , train_s2, train_y = load_dataset('train')
    val_s1 , val_s2, val_y = load_dataset('dev_matched')
    # test_s1 , test_s2, test_y = load_dataset('test')

    pickle.dump((train_s1, train_s2, train_y), open('data/multinli_train.pb', 'wb'))
    pickle.dump((val_s1, val_s2, val_y), open('data/multinli_val.pb', 'wb'))
    # pickle.dump((test_s1, test_s2, test_y), open('data/multinli_test.pb', 'wb'))

print('Finished loading dataset.')

class CBOW(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim, pretrained_embed_path, pad_idx):
        super(CBOW, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.l1 = nn.Linear(4*embedding_dim, args.hidden_dim)
        self.l2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.l3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.l4 = nn.Linear(args.hidden_dim, num_labels)
        self.relu = nn.ReLU()
        self.pad_idx = pad_idx

        # self.init_weights(pretrained_embed_path)

    def init_weights(self, pretrained_embed_path):
        # return
        self.emb.weight.data.copy_(pretrained_embed_path)
        self.emb.weight.requires_grad = False

    def forward(self, x1, x2):
        emb_out1 = self.emb(x1)
        mask1 = 1 - (x1 == self.pad_idx).float()
        batch_len = torch.sum(mask1, dim=1).unsqueeze(1)
        sentence_emb1 = torch.div(torch.sum(emb_out1, dim=1) , batch_len)

        emb_out2 = self.emb(x2)
        mask2 = 1 - (x2 == self.pad_idx).float()
        batch_len = torch.sum(mask2, dim=1).unsqueeze(1)
        sentence_emb2 = torch.div(torch.sum(emb_out2, dim=1) , batch_len)

        # sentence_emb = torch.cat((sentence_emb1, sentence_emb2), dim=1)
        sentence_emb = torch.cat((sentence_emb1, sentence_emb2, torch.abs(sentence_emb1-sentence_emb2), sentence_emb1*sentence_emb2), dim=1)

        out = self.l4(self.relu(self.l3(self.relu(self.l2(self.relu(self.l1(sentence_emb)))))))
        return out

cbow = CBOW(len(gold_label2idx.keys()), len(vocab), embedding_dim, syn_emb, w2idx['<pad>']).to(device)
optimizer = optim.Adam(cbow.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def pad_sequences(s):
    pad_token = w2idx['<pad>']
    # print(s)
    lengths = [len(s1) for s1 in s]
    longest_sent = max(lengths)
    padded_X = np.ones((args.batch_size, longest_sent), dtype=np.int64) * pad_token
    for i, x_len in enumerate(lengths):
        sequence = s[i]
        padded_X[i, 0:x_len] = sequence[:x_len]
    # print(padded_X)
    return padded_X


def evaluate(data_source):
    cbow.eval()
    total_loss = 0
    total_acc = 0

    data_s1 = data_source[0]
    data_s2 = data_source[1]
    data_y = data_source[2]

    num_iterations = len(data_y) // args.batch_size
    with torch.no_grad():
        for i in range(num_iterations):

            batch_s1 = pad_sequences(data_s1[i * args.batch_size : (i+1) * args.batch_size])
            batch_s2 = pad_sequences(data_s2[i * args.batch_size : (i+1) * args.batch_size])
            batch_y = data_y[i * args.batch_size : (i+1) * args.batch_size]

            batch_s1 = torch.LongTensor(batch_s1).to(device)
            batch_s2 = torch.LongTensor(batch_s2).to(device)
            batch_y = torch.LongTensor(batch_y).to(device)

            predictions = cbow(batch_s1, batch_s2)

            predictions_np = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
            acc = np.mean(predictions_np == batch_y.cpu().numpy())

            loss = criterion(predictions, batch_y)
            # print(loss)

            total_loss += loss.item()
            total_acc += acc

    curr_loss = total_loss / num_iterations
    curr_acc = total_acc / num_iterations

    print('|lr {:02.10f} | loss {:5.2f} | acc {:5.2f}' \
            .format(optimizer.param_groups[0]['lr'], curr_loss, curr_acc))

    return curr_loss, curr_acc

def train():
    for _ in range(args.epochs):
        cbow.train()
        total_loss = 0
        total_acc = 0
        patience = 0
        best_val_loss = float('Inf')
        # print(len(train_s1))

        start_time = time.time()

        num_iterations = len(train_y) // args.batch_size
        for i in range(num_iterations):
            optimizer.zero_grad()

            batch_s1 = pad_sequences(train_s1[i * args.batch_size : (i+1) * args.batch_size])
            batch_s2 = pad_sequences(train_s2[i * args.batch_size : (i+1) * args.batch_size])
            batch_y = train_y[i * args.batch_size : (i+1) * args.batch_size]

            batch_s1 = torch.LongTensor(batch_s1).to(device)
            batch_s2 = torch.LongTensor(batch_s2).to(device)
            batch_y = torch.LongTensor(batch_y).to(device)

            predictions = cbow(batch_s1, batch_s2)

            predictions_np = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
            # print(np.mean((predictions_np == batch_y.cpu().numpy()).astype(np.int64)))
            acc = np.mean(predictions_np == batch_y.cpu().numpy())

            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

            if i % args.log_interval == 0 and i > 0:
                curr_loss = total_loss / args.log_interval
                curr_acc = total_acc / args.log_interval
                elapsed = time.time() - start_time

                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | loss {:5.2f} | acc {:5.2f}' \
                        .format(_, i, num_iterations, optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, curr_loss, curr_acc))
                total_loss = 0
                total_acc = 0
                start_time = time.time()


        val_loss, val_acc = evaluate((val_s1, val_s2, val_y))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(cbow, 'cbow_model_emb.pb')
            print('Model Saved')
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print('Exiting from Training. Early Stopping. ')
                break


train()
