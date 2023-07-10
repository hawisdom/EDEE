from collections import Counter, defaultdict
import numpy as np
import torch
import os
import json
import gensim
from ltp import LTP
import pickle
from torch.utils.data import Dataset
from data_process import *
from bert_serving.client import BertClient

logger = logging.getLogger(__name__)

role_role2idx,idx2role_role = get_role_role2idx()
stopwords = get_stop_words()


def load_datasets_and_vocabs(args):
    train_example_file = os.path.join(args.cache_dir, 'train_example.pkl')
    test_example_file = os.path.join(args.cache_dir, 'test_example.pkl')
    train_weight_file = os.path.join(args.cache_dir, 'train_weight_catch.txt')
    test_weight_file = os.path.join(args.cache_dir, 'test_weight_catch.txt')

    if os.path.exists(train_example_file) and os.path.exists(test_example_file):
        logger.info('Loading train_example from %s', train_example_file)
        with open(train_example_file, 'rb') as f:
            train_examples = pickle.load(f)

        logger.info('Loading test_example from %s', test_example_file)
        with open(test_example_file, 'rb') as f:
            test_examples = pickle.load(f)

        with open(train_weight_file, 'rb') as f:
            train_labels_weight = torch.Tensor(json.load(f))
        with open(test_weight_file, 'rb') as f:
            test_labels_weight = torch.Tensor(json.load(f))
    else:
        train_file = os.path.join(args.dataset_path,'train.json')
        test_file = os.path.join(args.dataset_path,'test.json')
        user_dict_file = os.path.join(args.dataset_path,'company.txt')

        logger.info('Loading ltp tool')
        ltp = LTP()
        if not os.path.exists(user_dict_file):
            generate_user_dict([train_file,test_file],user_dict_file)
        ltp.init_dict(path=user_dict_file)

        logger.info('Creating train examples')
        train_examples,train_labels_weight = create_example(train_file,ltp,'train')
        logger.info('store train examples to cache file')
        with open(train_example_file, 'wb') as f:
            pickle.dump(train_examples, f, -1)

        logger.info('Creating test examples')
        test_examples, test_labels_weight = create_example(test_file, ltp,'test')
        logger.info('store test examples to cache file')
        with open(test_example_file, 'wb') as f:
            pickle.dump(test_examples, f, -1)

        with open(train_weight_file,'w') as wf:
            json.dump(train_labels_weight,wf)
        logger.info('Creating test_weight_cache')
        with open(test_weight_file,'w') as wf:
            json.dump(test_labels_weight,wf)

    logger.info('Train set size: %s', len(train_examples))
    logger.info('Test set size: %s,', len(test_examples))

    # Build word vocabulary(dep_tag, part of speech) and save pickles.
    word_vecs,word_vocab,wType_tag_vocab = load_and_cache_vocabs(train_examples+test_examples, args)

    embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
    args.token_embedding = embedding

    train_dataset = ED_Dataset(train_examples,args,word_vocab,wType_tag_vocab)
    test_dataset = ED_Dataset(test_examples,args,word_vocab,wType_tag_vocab)

    return train_dataset,train_labels_weight,test_dataset,test_labels_weight,word_vocab,wType_tag_vocab

def generate_user_dict(files,path):
    f = open(path, 'w', encoding='utf-8')
    for file in files:
        with open(file, 'r', encoding='utf-8-sig') as fp:
            datas = json.load(fp)

        for doc in datas:
            entities = doc[1]['ann_valid_mspans']
            for entity in entities:
                f.write(entity.strip()+'\n')
    f.close()

def create_example(file,ltp,dataset_type):
    with open(file, 'r', encoding='utf-8-sig') as fp:
        datas = json.load(fp)

    # {文档: {词语: 次数}}
    f = open('./data/repeat.json','r',encoding='utf-8-sig')
    repeat_info = json.load(f)
    examples = []
    label_ids = []
    for doc in datas:
        sentences = doc[1]['sentences']
        events = doc[1]['recguid_eventname_eventdict_list']
        arg_dranges = doc[1]['ann_mspan2dranges']
        mspan2guess_field = doc[1]['ann_mspan2guess_field']
        word_info_dict = {}
        for sent_idx,sentence in enumerate(sentences):
            sentence = sentence.strip('')
            if len(sentence) == 0:
                continue
            words, hidden = ltp.seg([sentence.strip()])
            words = words[0]
            word_loc = 0
            for word_idx,word in enumerate(words):
                type_flag = False
                word_info = {}
                for arg,dranges in arg_dranges.items():
                    for sent,ch_s,ch_e in dranges:
                        if sent == sent_idx and word_loc >= ch_s and word_loc <= ch_e:
                            if dataset_type == 'train':
                                word_info = get_word_info(sent_idx,events,word,mspan2guess_field[arg])
                            else:
                                word_info = get_word_info_test(sent_idx,events,word,mspan2guess_field[arg],repeat_info,doc[0])
                            type_flag = True
                            break
                    if type_flag:
                        break
                if (not type_flag) or (type_flag and not word_info[word]):
                    word_info[word] =[(None,None,sent_idx,word,None,None,'Other')]
                word_loc += len(word)
                if word not in word_info_dict.keys() and word not in stopwords:
                    word_info_dict.update(word_info)

        valid_words = []
        all_words = []
        word_id = 0
        for word,infos in word_info_dict.items():
            for info in infos:
                event_type, event_id,sent_idx, word, role, loc_in_arg,word_type = info
                if event_type is not None:
                    valid_words.append((word_id, event_type,event_id, role, loc_in_arg))
                all_words.append((sent_idx,word, word_type))
                word_id += 1

        arg_arg_adj = torch.zeros(len(all_words), len(all_words), requires_grad=False, dtype=torch.long)
        for w_id1,etype1,e_id1,role1,loc_in_arg1 in valid_words:
            for w_id2,etype2,e_id2,role2,loc_in_arg2 in valid_words:
                if w_id1 == w_id2:
                    continue
                if (etype1 == etype2) and (e_id1 == e_id2):
                    if loc_in_arg1 == 0:
                        r1 = 'B_' + role1
                    else:
                        r1 = 'I_' + role1
                    if loc_in_arg2 == 0:
                        r2 = 'B_' + role2
                    else:
                        r2 = 'I_' + role2
                    arg_arg_adj[w_id1][w_id2] = role_role2idx[(etype1,r1,r2)]

        example = {'words': [], 'sens': [], 'word_types': []}
        for sent_idx,word,word_type in all_words:
            example['words'].append(word)
            example['sens'].append(sentences[sent_idx])
            example['word_types'].append(word_type)

        example['role_role_adj'] = arg_arg_adj.reshape(-1).tolist()
        examples.append(example)

        for line in arg_arg_adj.tolist():
            for ele in line:
                label_ids.append(ele)

    f.close()
    label_weight = get_labels_weight(label_ids)
    return examples,label_weight


def get_word_info(sent_idx,events,word,word_type):
    word_info = {word:[]}
    word_key_info = []
    for event in events:
        event_id = event[0]
        event_type = event[1]
        role_args = event[2]
        for role,arg in role_args.items():
            if arg is None:
                continue
            if word in arg:
                loc_in_arg = arg.find(word)
                if (event_type,word,role) not in word_key_info:
                    word_key_info.append((event_type,word,role))
                    word_info[word].append((event_type,event_id, sent_idx, word, role, loc_in_arg,word_type))
    return word_info

#{文档: {词语: 次数}}
def get_word_info_test(sent_idx,events,word,word_type,repeat_info,doc_id):
    word_info = {word: []}
    if doc_id not in repeat_info.keys():
        return word_info
    repeat_word_info = repeat_info[doc_id]
    for i in range(int(repeat_word_info[word])):
        word_info[word].append([None,None,sent_idx,word,None,None,None])

    # for evaluation
    word_key_info = []
    for i,event in enumerate(events):
        if i > len(word_info[word]):
            continue
        event_id = event[0]
        event_type = event[1]
        role_args = event[2]
        for role,arg in role_args.items():
            if arg is None:
                continue
            if word in arg:
                loc_in_arg = arg.find(word)
                if (event_type,word,role) not in word_key_info:
                    word_key_info.append((event_type,word,role))
                    word_info[word][i] = [event_type,event_id, sent_idx, word, role, loc_in_arg,word_type]

    return word_info


def get_labels_weight(label_ids):
    nums_labels = Counter(label_ids)
    nums_labels = [(l,k) for k, l in sorted([(j, i) for i, j in nums_labels.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size//2-1][1])/2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weight_list = []
    # roles_lookup = {'none': 0, 'sub': 1, 'pred': 2, 'obj': 3}
    for value_id in role_role2idx.values():
        if value_id not in label_ids:
            weight_list.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weight_list.append(median/label[1])
                    break
    return weight_list

def load_and_cache_vocabs(examples,args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    embedding_cache_path = os.path.join(args.cache_dir, 'embedding')
    if not os.path.exists(embedding_cache_path):
        os.makedirs(embedding_cache_path)

    # Build or load word vocab and word2vec embeddings.
    cached_word_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_word_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vocab_file):
        logger.info('Loading word vocab from %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'rb') as f:
            word_vocab = pickle.load(f)
    else:
        logger.info('Creating word vocab from dataset %s',args.dataset_name)
        word_vocab = build_text_vocab(examples)
        logger.info('Word vocab size: %s', word_vocab['len'])
        logging.info('Saving word vocab to %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'wb') as f:
            pickle.dump(word_vocab, f, -1)

    cached_word_vecs_file = os.path.join(embedding_cache_path, 'cached_{}_word_vecs.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vecs_file):
        logger.info('Loading word vecs from %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'rb') as f:
            word_vecs = pickle.load(f)
    else:
        logger.info('Creating word vecs from %s', args.embedding_dir)
        word_vecs = load_bert_embedding(word_vocab['itos'])
        logger.info('Saving word vecs to %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'wb') as f:
            pickle.dump(word_vecs, f, -1)

    # Build vocab of word type tags.
    cached_wType_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_wType_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_wType_tag_vocab_file):
        logger.info('Loading vocab of word type tags from %s', cached_wType_tag_vocab_file)
        with open(cached_wType_tag_vocab_file, 'rb') as f:
            wType_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of word type tags.')
        wType_tag_vocab = build_wType_tag_vocab(examples, min_freq=0)
        logger.info('Saving word type tags  vocab, size: %s, to file %s', wType_tag_vocab['len'], cached_wType_tag_vocab_file)
        with open(cached_wType_tag_vocab_file, 'wb') as f:
            pickle.dump(wType_tag_vocab, f, -1)

    return word_vecs,word_vocab,wType_tag_vocab

def load_bert_embedding(word_list):
    word_vectors = []
    bc = BertClient()
    for word in word_list:
        if word == 'pad':
            word_vectors.append(np.zeros(768, dtype=np.float32))
        else:
            word_vectors.append(bc.encode([word])[0])
    return word_vectors


def _default_unk_index():
    return 1

def build_text_vocab(examples, vocab_size=1000000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['words'])

    itos = ['pad']
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_wType_tag_vocab(examples, vocab_size=1000, min_freq=0):

    counter = Counter()
    for example in examples:
        counter.update(example['word_types'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


class ED_Dataset(Dataset):
    def __init__(self, examples,args,word_vocab,wType_tag_vocab):
        self.examples = examples
        self.args = args
        self.word_vocab = word_vocab
        self.wType_tag_vocab = wType_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        items = e['word_ids'],e['wType_ids'],e['role_role_adj']

        # items_tensor = tuple(torch.tensor(t) for t in items)
        return items

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.examples)):
            self.examples[i]['word_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['words']]
            self.examples[i]['wType_ids'] = [self.wType_tag_vocab['stoi'][t] for t in self.examples[i]['word_types']]


def my_collate(batch):
    '''
    Pad event in a batch.
    Sort the events based on length.
    Turn all into tensors.
    '''
    # from Dataset.__getitem__()
    word_ids,wType_ids,labels  = zip(
        *batch)  # from Dataset.__getitem__()

    word_ids = torch.tensor(word_ids[0])
    wType_ids = torch.tensor(wType_ids[0])
    labels = torch.tensor(labels[0])

    return word_ids,wType_ids,labels
