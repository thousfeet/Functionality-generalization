import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self,  ratio, root, language, loop, type):
        self.ratio = ratio
        self.root = root
        self.language = language
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None
        self.loop = loop
        self.type = type

    # parse source code
    def parse_source(self, output_file, option):
        path = self.root+self.language+'/'+output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
            if self.language is 'c':
                from pycparser import c_parser
                parser = c_parser.CParser()
                source = pd.read_pickle(self.root+self.language+'/programs.pkl')
                source.columns = ['id', 'code', 'label']
                source['code'] = source['code'].apply(parser.parse)
                source.to_pickle(path)
            else:
                import javalang
                def parse_program(func):
                    tokens = javalang.tokenizer.tokenize(func)
                    parser = javalang.parser.Parser(tokens)
                    tree = parser.parse_member_declaration()
                    return tree
                source = pd.read_csv(self.root+self.language+'/bcb_funcs_all.tsv', sep='\t', header=None, encoding='utf-8')
                source.columns = ['id', 'code']
                source['code'] = source['code'].apply(parse_program)
                source.to_pickle(path)
        self.sources = source
        return source

    # create clone pairs
    def read_pairs(self, filename):
        pairs = pd.read_pickle(self.root+self.language+'/'+filename)
        self.pairs = pairs

    # split data for training, developing and testing
    def split_data(self, loop):
        data_path = self.root+self.language+'/'
        data = self.pairs
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=666)
        # train = data.iloc[:train_split]
        train = pd.read_pickle(data_path + "pickles/{}_{}_train.pkl".format(train_id1, train_id2))
        for i in range(1, loop+1):
            samples = pd.read_pickle(data_path + "pickles/" + self.type +"_{}_{}_{}_{}.pkl".format(train_id1, test_id1, K, i))
            # samples = samples.sample(n=20000, replace=True)
            train = train.append(samples, ignore_index=True)

        # dev = data.iloc[train_split:val_split]
        dev = pd.read_pickle(data_path + "pickles/{}_{}_train.pkl".format(test_id1, test_id2))

        # test = data.iloc[val_split:]
        test = pd.read_pickle(data_path + "pickles/{}_{}_test.pkl".format(test_id1, test_id2))

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = data_path+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = data_path+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = data_path+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root+self.language+'/'
        if not input_file:
            input_file = self.train_file_path
        pairs = pd.read_pickle(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id',drop=False).loc[train_ids]
        if not os.path.exists(data_path+'train/embedding'):
            os.mkdir(data_path+'train/embedding')
        if self.language is 'c':
            sys.path.append('../')
            from prepare_data import get_sequences as func
        else:
            from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data import get_blocks as func
        else:
            from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.DataFrame(self.sources, copy=True)
        trees['code'] = trees['code'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self,data_path,part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1,inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root+self.language+'/'+part+'/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('read id pairs...')
        if self.language is 'c':
            self.read_pairs('oj_clone_ids.pkl')
        else:
            self.read_pairs('bcb_pair_ids.pkl')
        print('split data...')
        self.split_data(self.loop)
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')


import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    id1, id2, x1, x2, labels = [], [], [], [], []
    for _, item in tmp.iterrows():
        id1.append(item['id1'])
        id2.append(item['id2'])
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels), id1, id2


def train(lang, loop, K, TYPE):
    categories = 1
    if lang == 'java':
        categories = 5
    print("Train for ", str.upper(lang))
    train_data = pd.read_pickle('data/c/train/blocks.pkl').sample(frac=1)
    test_data = pd.read_pickle('data/c/test/blocks.pkl')

    word2vec = Word2Vec.load("data/c/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 32
    USE_GPU = True

    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    # print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    for t in range(1, categories+1):
        if lang == 'java':
            train_data_t = train_data[train_data['label'].isin([t, 0])]
            train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

            test_data_t = test_data[test_data['label'].isin([t, 0])]
            test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
        else:
            train_data_t, test_data_t = train_data, test_data
        # training procedure
        for epoch in range(EPOCHS):
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels, _,_ = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output, _, _ = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

        print("Testing-%d..."%t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels,_,_ = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output, _, _ = model(test1_inputs, test2_inputs)
            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)
        if lang == 'java':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    torch.save(model.state_dict(), "models/"+TYPE+"_{}_{}_{}.pt".format(K, train_id1, test_id1))

    #sampling
    print("sampling start...")

    actives = pd.read_pickle('data/c/dev/blocks.pkl') #.sample(frac=1)

    predicts = []
    trues = []
    total_loss = 0.0
    total = 0.0
    i = 0
    inputs1, inputs2 = [], []
    logits, embds = [], []

    while i < len(actives):
        batch = get_batch(actives, i, BATCH_SIZE)
        i += BATCH_SIZE
        test1_inputs, test2_inputs, test_labels, id1, id2 = batch

        inputs1 += id1
        inputs2 += id2

        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output, embd1, embd2 = model(test1_inputs, test2_inputs)
        # embd = torch.cat([embd1, embd2])

        logits.extend((abs(output.data-0.5)).cpu().numpy())
        # embds.extend(embd.data.cpu().numpy())

        loss = loss_function(output, Variable(test_labels))

        # calc testing acc
        predicted = (output.data > 0.5).cpu().numpy()
        predicts.extend(predicted)
        trues.extend(test_labels.cpu().numpy())
        total += len(test_labels)
        total_loss += loss.item() * len(test_labels)

    logits = np.array([x[0] for x in logits])
    # embds = np.array(embds)
    inputs1 = np.array(inputs1)
    inputs2 = np.array(inputs2)
    labels = np.array([x[0] for x in trues])
    # print(labels[:10])

    # K highest uncertain pairs
    idx = np.argpartition(logits, K)
    logits = logits[idx[:K]]
    # embds = embds[idx[:K]]
    pair_nodes = np.array(list(zip(inputs1[idx[:K]], inputs2[idx[:K]], labels[idx[:K]])))

    df = pd.DataFrame(pair_nodes, columns=['id1', 'id2', 'label'])
    print(df.shape)
    print(df[df['label'] == 0].shape[0] / df.shape[0])
    df.to_pickle("data/c/pickles/"+TYPE+"_{}_{}_{}_{}.pkl".format(train_id1, test_id1, K, loop))
    print(df.head())

    #M clusters, N samples
    # M, N = 100, 1000
    # kmeans = KMeans(n_clusters=M, random_state=666).fit(embds)
    # cluster_idx = kmeans.labels_
    # result = []
    # for cls in range(M):
    #     cls_nodes = pair_nodes[cluster_idx == cls]
    #     select = np.random.choice(list(range(len(cls_nodes))), round(len(cls_nodes)/K * N), False)
    #     result += [pair_nodes[i] for i in select]
    #     # top = np.argmin(logits[cluster_idx == cls])
    #     # result.append(pair_nodes[top])
    #
    # df = pd.DataFrame(result, columns=['id1', 'id2', 'label'])
    # print(df.shape)
    # print(df[df['label'] == 0].shape[0] / df.shape[0])
    # df.to_pickle("data/c/cluster_2.pkl")
    # print(df.head())

    return f1


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
parser.add_argument('--lang')
parser.add_argument('--type')
parser.add_argument('--train_low', type=int)
parser.add_argument('--train_up', type=int)
parser.add_argument('--test_low', type=int)
parser.add_argument('--test_up', type=int)
parser.add_argument('--k', type=int)
parser.add_argument('--loop', type=int)


args = parser.parse_args()
if not args.lang:
    print("No specified dataset")
    exit(1)

train_id1, train_id2 = args.train_low, args.train_up
test_id1, test_id2 = args.test_low, args.test_up
K = args.k
TYPE = args.type
LOOP = args.loop
F1 = []

for loop in range(LOOP+1):
    print("loop {}:".format(loop))
    ppl = Pipeline('3:1:1', 'data/', str(args.lang), loop, TYPE)
    ppl.run()
    f1 = train(args.lang, loop+1, K, TYPE)
    F1.append(f1)

print("f1:", F1)
