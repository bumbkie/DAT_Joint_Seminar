import gluonnlp as nlp
from sklearn.model_selection import train_test_split
from dataframe_pre import term_explain_preprocessing 
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Dataset 자료 형식 : [[용어1, 설명1], [용어1, 설명2], ... , [용어2, 설명1], ...] 형식의 리스트
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

def make_dataset(label_mapping, train, test):
    #BERTDataset에 넣기 적절한 모양으로 바꿈
    dataset_train = []
    dataset_test = []

    lab = list(train.iloc[:,0])
    sen = list(train.iloc[:,1])

    for l,s in zip(lab, sen):
      dataset_train.append([label_mapping[l],s])

    lab2 = list(test.iloc[:,0])
    sen2 = list(test.iloc[:,1])

    for l,s in zip(lab2, sen2):
      dataset_test.append([label_mapping[l],s])

    return dataset_train, dataset_test

def make_dataloader(csv_file):
    #nlp module의 BERTVocab을 사용할 vocab으로 지정
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
    tok= tokenizer.tokenize

    df, label_mapping, result_dict = term_explain_preprocessing(csv_file)
    train, test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 42, stratify = df['용어'])

    data_train = BERTDataset(dataset_train, 1, 0, tok, vocab, max_len, True, False)
    data_test = BERTDataset(dataset_test, 1, 0, tok, vocab, max_len, True, False)

    train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)

    return train_dataloader, test_dataloader
