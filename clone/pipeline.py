import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import defaultdict

dicttChar = {}
listchar = ['!', '%', '&', '+', '*', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', '=', '<', '>',
            'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'J', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T',
            'W', 'V', 'Y', 'X', '[', 'Z', ']', '_', '^', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm',
            'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z', '|', '~']

for i in range(0, len(listchar)):
    dicttChar[listchar[i]] = [1.0 if j == i else 0.0 for j in range(len(listchar))]

def getWordEmd(word):
    listrechar = np.array([0.0 for i in range(0, len(listchar))])
    tt = 1
    for lchar in word:
        if lchar not in listchar:
            continue
        listrechar += np.array(((len(word) - tt + 1) * 1.0 / len(word)) * np.array(dicttChar[lchar]))
        tt += 1
    return listrechar


class Pipeline:
    def __init__(self,  ratio, root, language, train_low, train_up, test_low, test_up):
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
        self.pace_dic = defaultdict(int)
        self.pace_embd = {}
        self.train_low, self.train_up, self.test_low, self.test_up = train_low, train_up, test_low, test_up

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
    def split_data(self):
        data_path = self.root+self.language+'/'
        data = self.pairs
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=666)
        # train = data.iloc[:train_split]
        train = pd.read_pickle(data_path + "pickles/{}_{}_train.pkl".format(self.train_low, self.train_up))
        # train = train[:20000]
        # samples = pd.read_pickle(data_path + "uncertain_1_15_100_1.pkl")
        # samples = samples.append(pd.read_pickle(data_path + "uncertain_500_2.pkl"), ignore_index=True)
        # samples = samples.append(pd.read_pickle(data_path + "uncertain_500_3.pkl"), ignore_index=True)
        # samples = samples.append(pd.read_pickle(data_path + "uncertain_500_4.pkl"), ignore_index=True)
        # samples = samples.append(pd.read_pickle(data_path + "16_30_cluster_250_5.pkl"), ignore_index=True)
        # samples = samples.sample(n=20000, replace=True)
        # train = train.append(samples, ignore_index=True)

        # dev = data.iloc[train_split:val_split]
        dev = pd.read_pickle(data_path + "pickles/{}_{}_test.pkl".format(self.test_low, self.test_up))

        # test = data.iloc[val_split:]
        test = pd.read_pickle(data_path + "pickles/{}_{}_test.pkl".format(self.test_low, self.test_up))

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

        trees = self.sources.set_index('id',drop=False) #.loc[train_ids]

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
        # print(str_corpus)
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        # from gensim.models.word2vec import Word2Vec
        # w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        # w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

        # trees = pd.DataFrame(self.sources, copy=True)
        # print(trees['code'])

        now = 1
        for sens in str_corpus:
            for x in sens.split():
                # print(self.pace_dic)
                if x not in self.pace_dic:
                    self.pace_dic[x]=now
                    self.pace_embd[now] = getWordEmd(x)
                    now+=1
        self.pace_embd[0] = np.array([0.0 for i in range(0, len(listchar))])
        # if "FuncDef" in self.pace_dic:
        #     print("FuncDef in")
        #     print(self.pace_embd[self.pace_dic["FuncDef"]])

        pace_embds = np.zeros((len(self.pace_embd), len(listchar)))
        for x in self.pace_embd:
            pace_embds[x] = self.pace_embd[x]
        np.save("data/c/train/embedding/pace_embd_new.npy", pace_embds)
        # print(pace_embds)

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data import get_blocks as func
        else:
            from utils import get_blocks_v1 as func
        # from gensim.models.word2vec import Word2Vec

        # word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        # vocab = word2vec.vocab
        # max_token = word2vec.syn0.shape[0]

        dic = self.pace_dic
        def tree_to_index(node):
            token = node.token
            # print(token)
            result = [dic[token]]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            # print(result)
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                # print(b)
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
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')
        print("Done")


import argparse
parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
parser.add_argument('--lang')
parser.add_argument('--train_low', type=int)
parser.add_argument('--train_up', type=int)
parser.add_argument('--test_low', type=int)
parser.add_argument('--test_up', type=int)

args = parser.parse_args()
if not args.lang:
    print("No specified dataset")
    exit(1)
ppl = Pipeline('3:1:1', 'data/', str(args.lang), args.train_low, args.train_up, args.test_low, args.test_up)
ppl.run()


