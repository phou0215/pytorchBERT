# 2021-04-22 Pytorch transformers sentence classification
# -*- coding: utf-8 -*-

import sys
import os
import torch
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from konlpy.tag import Okt
from konlpy.tag import Komoran
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification,\
    BertConfig, BertTokenizerFast, DistilBertTokenizer,  DistilBertForSequenceClassification, DistilBertConfig
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
# from torch.optim import optimizer, Adam

# data set control
class PytorchDistilBERT():

    def __init__(self):
        super(PytorchDistilBERT, self).__init__()
        # main test data
        self.df_data = None

        # input 산툴물 정리
        self.list_memo = None
        self.list_label = None
        self.list_special = None
        self.list_rmstring = None
        self.list_corpus = None

        # split train and test data
        self.Train_Data_X = None
        self.Train_Data_Y = None
        self.Test_Data_X = None
        self.Test_Data_Y = None
        self.df_data = None
        self.label_index = None

        # type '' => base no customer tokenizer, type 'word' => word piece, type 'sentence' => sentence piece
        self.tokenizer_type = 'word'
        # if tokenizer_type is not default select konlpy parse type 'okt' or default is Okt , 'komoran' is Komoran
        self.subword_type = 'okt'
        # public tokenizer
        self.konlpy_parser = None

        # file_path
        self.model_root_dir = './save_models'
        self.vocab_root_dir = './save_vocab'
        self.file_path = './voc_data.xlsx'
        self.corpus_path = './' + self.subword_type + '_text.txt'
        self.corpus_raw_data_path = "./corpus.csv"
        self.model_dir = ''
        self.vocab_dir = ''

        # pyplot
        self.figure = None
        self.line_loss = None
        self.line_score = None

        # Options and utility
        self.model = None
        self.test_rate = 0.2
        self.batch_size = 5
        self.epoch = 10
        self.learning_rate = 1e-5
        self.device = torch.device('cpu')
        self.pattern = re.compile("([1-9]{1,2}\.)")

        self.stopString = ["안내", "여부", "사항", "장비", "확인", "원클릭", "품질", "후", "문의", "이력", "진단", "부탁드립니다.",
                           "증상", "종료", "문의", "양호", "정상", "고객", "철회", "파이", "특이", "간다", "내부", "외부", "권유",
                           "성향", "하심", "해당", "주심", "고함", "초기", "무관", "반려", "같다", "접수", " 무관", "테스트", "연락",
                           "바로", "처리", "모두", "있다", "없다", "하다", "드리다", "않다", "되어다", "되다", "부터", "예정", "드리다",
                           "해드리다", "신내역", "현기", "가신", 'ㅜ', "ㅠ"]

    # console 프린트 함수
    def setPrint(self, text):

        current = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        print("{}:\n{}".format(current, text) + "\n")

    # 현재 시간 구하여 3가지 타입으로 list return
    def getCurrent_time(self):

        nowTime = datetime.now()
        nowtime_str = nowTime.strftime('%Y-%m-%d %H:%M:%S')
        nowtime_str_2 = nowTime.strftime('%Y-%m-%d %H %M %S')
        return [nowTime, nowtime_str, nowtime_str_2]

    # GPU 사용 가능 확인 함수
    def check_cuda(self):

        cuda_flag = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_flag else 'cpu')
        self.setPrint('Training device : {}'.format(self.device))

        torch.manual_seed(7)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(7)

    # label 인덱스 부여하기
    def set_label_index(self, df_data, col_name='label'):

        labels = df_data[col_name].unique()
        label_dict = {}
        for index, label in enumerate(labels):
            label_dict[label] = index
        return label_dict

    # unless exists saved model directory, generate model's directory
    def generate_model_directory(self):

        # root dir ath check and generate
        if not os.path.isdir(self.model_root_dir):
            os.makedirs(self.model_root_dir, exist_ok=True)

        # generate models directory
        self.model_dir = '/BERT_TRAINING_MODEL_' + self.getCurrent_time()[2] + '/'
        os.makedirs(self.model_root_dir + self.model_dir, exist_ok=True)

    # ###################### matplot graph 지원함수 #######################

    def generate_graph(self):

        x = np.linspace(1, self.epoch, self.epoch)
        self.figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.set_title('LOSS')
        ax1.set(xlabel='Epoch', ylabel='avg_loss')
        self.line_loss, = ax1.plot(x, np.array([0.000] * self.epoch), 'tab:blue')

        ax2.set_title('F1 SCORE')
        ax2.set(xlabel='Epoch', ylabel='F1 Score')
        self.line_score, = ax2.plot(x, np.array([0.000] * self.epoch), 'tab:green')

    def init_graph(self):

        self.figure.canvas.draw()
        # self.figure.canvas.flush_events()

    def grid_loss_graph(self, y):
        x_data = np.linspace(1, self.epoch, self.epoch)
        self.line_loss.set_xdata(x_data)
        self.line_loss.set_ydata(y)

    def grid_f1_graph(self, y):
        x_data = np.linspace(1, self.epoch, self.epoch)
        self.line_score.set_xdata(x_data)
        self.line_score.set_ydata(y)

    # ####################################################################

    # ####################### 텍스트 전처리 지원함수 #########################

    # 특정 문자 구간 Parsing 함수(앞에서부터)
    def find_between(self, s, first, last):

        try:
            start = s.index(first) + len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""

    # 특정 문자 구간 Parsing 함수(뒤에서부터)
    def find_between_r(self, s, first, last):

        try:
            start = s.rindex(first) + len(first)
            end = s.rindex(last, start)
            return s[start:end]
        except ValueError:
            return ""

    # remove special char and confusing words united only one expression
    def remove_special(self, text):

        # 영문 모두 소문자로 변경
        text_data = text.lower()
        # 전화번호 모두 'tel로 치환'
        text_data = re.sub(r'\d{2,3}[-\.\s]*\d{3,4}[-\.\s]*\d{4}(?!\d)', 'tel', text_data)
        # 화폐는 'money'로 치환
        text_data = re.sub(r'\d{1,3}[,\.]\d{1,3}[만\천]?\s?[원]|\d{1,5}[만\천]?\s?[원]', 'money', text_data)
        text_data = re.sub(r'일/이/삼/사/오/육/칠/팔/구/십/백][만\천]\s?[원]', 'money', text_data)
        text_data = re.sub(r'(?!-)\d{2,4}[0]{2,4}(?!년)(?!.)|\d{1,3}[,/.]\d{3}', 'money', text_data)
        text_data = re.sub(r'[1-9]g', ' cellular ', text_data)
        text_data = re.sub(r'(유심|usim|sim|esim)', 'usim', text_data)
        text_data = re.sub(r'(sms|mms|메시지)', 'message', text_data)
        text_data = re.sub(r'통신.?내역', 'list', text_data)
        # web 주소는 'url'로 변경
        text_data = re.sub(
            r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
            'url',
            text_data)
        # 그 외의 특수문자는 모두 삭제
        text_data = re.sub(r'[-=+,_#/\?^$@*\"※~&%ㆍ!』\‘|\(\)\[\]\<\>\{\}`><\':;■◈▶●★☎]', ' ', text_data)
        # 앞서 list_rmstring 선언된 단어들 모두 제거
        for item in self.list_rmstring:
            text_data = text_data.replace(item, "")
        # 필수 제거 단어 제거
        for item in self.stopString:
            text_data = text_data.replace(item, "")

        # 앞 뒤 공백 제거
        text_data = text_data.strip()
        return text_data

    # konlpy text parsing 실행 함수
    def subword_text(self, text):
        try:
            mal_ist = self.konlpy_parser.morphs(text)
            return mal_ist
        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))
            return None

    # konlpy 형태소 분류 형태 텍스트 전처리
    def text_konlpy_filter(self, list_doc, list_label):

        try:
            list_return_docs = []
            list_return_labels = []
            for idx, doc in enumerate(list_doc):
                for item in self.list_special:
                    if item in doc:
                        spec_string = item
                        doc = doc + "/e"
                        text = self.find_between(doc, spec_string, "/e")
                        if self.pattern.search(text):
                            pattern_list = self.pattern.findall(text)
                            doc = self.find_between(doc, spec_string, pattern_list[0])
                        else:
                            doc = doc.replace("/e", "")
                        break
                doc = self.remove_special(doc)
                if doc == "":
                    continue
                split_doc = self.subword_text(doc)
                doc = " ".join([s.strip() for s in split_doc if s])
                list_return_docs.append(doc)
                if list_label:
                    list_return_labels.append(list_label[idx])
            return list_return_docs, list_return_labels
        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))
            return None

    # 특수문자 혹은 치환만 하는 텍스트 전처리
    def text_normal_filter(self, list_doc, list_label):

        try:
            list_return_docs = []
            list_return_labels = []
            for idx, doc in enumerate(list_doc):
                for item in self.list_special:
                    if item in doc:
                        spec_string = item
                        doc = doc + "/e"
                        text = self.find_between(doc, spec_string, "/e")
                        if self.pattern.search(text):
                            pattern_list = self.pattern.findall(text)
                            doc = self.find_between(doc, spec_string, pattern_list[0])
                        else:
                            doc = doc.replace("/e", "")
                        break
                doc = self.remove_special(doc)
                if doc == "":
                    continue
                list_return_docs.append(doc)
                if list_label:
                    list_return_labels.append(list_label[idx])

            return list_return_docs, list_return_labels
        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))
            return None

    # ####################################################################

    # ####################### Tokenizer 생성 및 데이터 생성 지원함수 #########################

    def generate_corpus_file(self, list_text):

        try:
            with open(self.corpus_path, 'w', encoding='utf-8') as f:
                for line in list_text:
                    f.write(line+'\n')
            self.setPrint('Corpus File is created')
        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))

    # generate tokenizer
    def generate_custom_vocab(self):

        try:
            tokenizer = None
            # root dir path check and generate
            if not os.path.isdir(self.vocab_root_dir):
                os.makedirs(self.vocab_root_dir, exist_ok=True)

            # generate models directory
            self.vocab_dir = '/BERT_TRAINING_VOCAB_' + self.getCurrent_time()[2] + '/'
            os.makedirs(self.vocab_root_dir + self.vocab_dir, exist_ok=True)

            user_defined_symbols = ['[BOS]', '[EOS]', '[UNK]', '[UNK1]', '[UNK2]', '[UNK3]', '[UNK4]', '[UNK5]',
                                    '[UNK6]', '[UNK7]', '[UNK8]', '[UNK9]']
            unused_token_num = 200
            unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
            user_defined_symbols = user_defined_symbols + unused_list

            if self.tokenizer_type == 'word':
                # if lowercase is False must set strip_accents option as 'False'
                tokenizer = BertWordPieceTokenizer(strip_accents=False,
                                                   lowercase=True,
                                                   clean_text=True,
                                                   handle_chinese_chars=True,
                                                   wordpieces_prefix="##"
                                                   )
            else:
                tokenizer = SentencePieceBPETokenizer()
            # when selected 'base' going to use bert-base-uncased tokenizer... close function
            # training vocab start
            corpus_file = [self.corpus_path]
            vocab_size = 32000
            limit_alphabet = 6000
            min_frequency = 3
            tokenizer.train(files=corpus_file,
                            vocab_size=vocab_size,
                            special_tokens=user_defined_symbols,
                            min_frequency=min_frequency,  # 단어의 최소 발생 빈도, 3
                            limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
                            show_progress=True)

            self.setPrint('Customer Tokenizer Training is completed')

            sentence = '전화 통화가 정상적으로 안됨.'
            output = tokenizer.encode(sentence)
            self.setPrint('Tokenizer 테스트 문장: {}'.format(sentence))
            self.setPrint('Tokenizer 분석 결과\n=>idx: {}\n=>tokens: {}\n=>offset: {}\n=>decode: {}\n'.
                          format(output.ids, output.tokens, output.offsets, tokenizer.decode(output.ids)))

            # save tokenizer
            tokenizer.save_model(self.vocab_root_dir + self.vocab_dir)

            if self.tokenizer_type == 'sentence':

                list_vocab = []
                with open(self.vocab_root_dir + self.vocab_dir + '/merges.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() != '':
                            list_vocab.append(line)
                print('출력내용 {}'.format(list_vocab[len(list_vocab) - 1]))
                with open(self.vocab_root_dir + self.vocab_dir + '/merges.txt', 'w', encoding='utf-8') as f:
                    for vocab in list_vocab:
                        f.write(vocab)
                self.setPrint('Rewrite merges text file completed')

        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))

    # Read excel file and generate dataframe, list
    def generate_data(self):

        try:
            # input excel 읽어서 data frame으로 변환
            self.df_data = pd.read_excel(self.file_path, sheet_name="통품전체VOC", index_col=None)
            if self.df_data.shape:
                self.setPrint('INPUT DATA SHAPE : {}'.format(self.df_data.shape))

            df_config1 = pd.read_excel(self.file_path, sheet_name="예약어리스트", index_col=None)
            df_config2 = pd.read_excel(self.file_path, sheet_name="Stop word", index_col=None)

            # N/A 값 drop
            self.df_data = self.df_data.dropna()
            df_config1 = df_config1.dropna()
            df_config2 = df_config2.dropna()

            # 예약어 설정 값 선택
            self.list_special = df_config1['Special예약어'].tolist()
            # 단어 제거 형식 선택
            self.list_rmstring = df_config2['일반형식'].tolist()

            if self.tokenizer_type != '':

                if self.subword_type == 'okt':
                    self.konlpy_parser = Okt()
                else:
                    self.konlpy_parser = Komoran()
                # corpus data generate
                df_vocab_data = pd.read_csv(self.corpus_raw_data_path, encoding='CP949')
                self.list_corpus = df_vocab_data['메모'].tolist()
                # corpus data cleaning data
                self.list_corpus, list_labels = self.text_konlpy_filter(self.list_corpus, None)
                self.generate_corpus_file(self.list_corpus)

            # 메모분류 문자열 전처리
            self.df_data['메모분류'] = self.df_data['메모분류'].apply(lambda x: x.strip())
            # print 데이터 분포 확인
            self.setPrint('데이터 분포: \n{}'.format(self.df_data['메모분류'].value_counts()))

            # set the 'label' column  by '메모분류' index
            self.label_index = self.set_label_index(self.df_data, col_name='메모분류')
            self.df_data['label'] = self.df_data['메모분류'].replace(self.label_index)
            self.setPrint('Index Of Labels: \n{}'.format(self.label_index))

            # data frame index 번호 reset
            self.df_data.reset_index(drop=True)
            # get list_label
            self.list_label = self.df_data['label'].tolist()

            if self.tokenizer_type != '':
                self.list_memo, self.list_label = self.text_konlpy_filter(self.df_data['메모'].tolist(), self.list_label)
            else:
                self.list_memo, self.list_label = self.text_normal_filter(self.df_data['메모'].tolist(), self.list_label)

            self.Train_Data_X, self.Test_Data_X, \
            self.Train_Data_Y, self.Test_Data_Y = train_test_split(self.list_memo,
                                                                   self.list_label,
                                                                   test_size=self.test_rate,
                                                                   random_state=42,
                                                                   shuffle=True,
                                                                   stratify=self.list_label)
            self.setPrint('Train_Data_Rate: {:.2f}%, Test_Data_Rate: {:.2f}% '.
                          format((1 - self.test_rate) * 100, self.test_rate * 100))
            self.setPrint('Train_Data_Count: {}\nTest_Data_Count:{}\nCorpus_Data_Count:{}'.format(
                len(self.Train_Data_Y), len(self.Test_Data_Y), len(self.list_corpus)))
        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))

    # ####################################################################################

    # ####################### matplot graph 지원함수 #########################
    def f1_score_func(self, preds, labels):

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')

    def evaluate(self, dataloader_val):

        self.model.eval()
        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        return loss_val_avg, predictions, true_vals

    # ######################################################################

    # ####################### main 실행 함수 #########################

    def run(self):

        torch.cuda.empty_cache()

        # raw data polling and pretreatment datas
        self.generate_data()

        # generate save model directory
        self.generate_model_directory()

        if self.tokenizer_type != '':
            # generate corpus by Okt konlpy
            # self.generate_custom_morphs(self.list_memo)

            # generate tokenizer model
            self.generate_custom_vocab()

        tokenizer = None
        if self.tokenizer_type == '':
            # base tokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                      lowercase=True,
                                                      strip_accents=False,
                                                      local_files_only=True)
        elif self.tokenizer_type == 'word':
            # word piece tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(self.vocab_root_dir + "/BERT_TRAINING_VOCAB_2021-06-03 16 09 57",
                                                            strip_accents=False,
                                                            lowercase=True)
        else:
            # sentence tokenizer
            tokenizer = SentencePieceBPETokenizer(self.vocab_root_dir + self.vocab_dir + '/vocab.json',
                                                  self.vocab_root_dir + self.vocab_dir + '/merges.txt', )

        self.setPrint('Load Customer Vocab size : {}'.format(tokenizer.vocab_size))
        # tokenizer Loading check
        # tokenized_input_for_pytorch = tokenizer_for_load("i am very happy now", return_tensors="pt")
        # encoded_text = tokenizer("전화 통화가 정상적으로 안됨", return_tensors="pt")
        # self.setPrint("Tokens Text List: {}".format(
        #     [tokenizer.convert_ids_to_tokens(s) for s in encoded_text['input_ids'].tolist()[0]]))
        # self.setPrint("Tokens IDX  List: {}".format(encoded_text['input_ids'].tolist()[0]))
        # self.setPrint("Tokens Mask List: {}".format(encoded_text['attention_mask'].tolist()[0]))

        # transformed train data
        encoded_data_train = tokenizer.batch_encode_plus(
            self.Train_Data_X,
            add_special_tokens=True,
            return_attention_mask=True,
            # padding='longest',
            padding=True,
            max_length=256,
            return_tensors='pt',
            truncation=True
        )
        # transformed validation data
        encoded_data_val = tokenizer.batch_encode_plus(
            self.Test_Data_X,
            add_special_tokens=True,
            return_attention_mask=True,
            # padding='longest',
            padding=True,
            max_length=256,
            return_tensors='pt',
            truncation=True
        )

        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(self.Train_Data_Y)

        input_ids_test = encoded_data_val['input_ids']
        attention_masks_test = encoded_data_val['attention_mask']
        labels_test = torch.tensor(self.Test_Data_Y)

        dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

        # local_files_only = True
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                         num_labels=len(self.label_index),
                                                                         output_attentions=False,
                                                                         output_hidden_states=False,
                                                                         local_files_only=True).to(self.device)

        # dataLoader
        dataloader_train = DataLoader(dataset_train,
                                      sampler=RandomSampler(dataset_train),
                                      batch_size=self.batch_size,
                                      drop_last=True)

        dataloader_test = DataLoader(dataset_test,
                                     sampler=RandomSampler(dataset_test),
                                     batch_size=self.batch_size)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(dataloader_train) * self.epoch)

        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
        #                                                                num_warmup_steps=0,
        #                                                                num_training_steps=len(dataloader_train) * self.epoch)
        # for loss f1 graph
        total_loss = np.array([0.0000] * 5)
        total_score = np.array([0.0000] * 5)

        # Training start
        for epoch in range(1, self.epoch + 1):
            self.setPrint('Start of Epoch {}'.format(epoch))
            self.model.train()
            loss_train_total = 0

            for idx, batch in enumerate(dataloader_train):
                self.model.zero_grad()
                batch = tuple(b.to(self.device) for b in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          }
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if idx % 100 == 0:
                    self.setPrint('[{}]Epoch {}/{} training_loss : {:.4f}'.format(epoch, idx, len(dataloader_train),
                                                                                  loss.item() / len(batch)))
                # gpu memory reset
                batch = None
                torch.cuda.empty_cache()

            # model save
            torch.save(self.model.state_dict(),
                       self.model_root_dir + self.model_dir + 'BERT_dict_epoch_{}.model'.format(epoch))
            self.setPrint('Save fine_tuned_BERT_epoch_{}.model'.format(epoch))
            self.setPrint('\nEnd of Epoch {}'.format(epoch))

            loss_train_avg = loss_train_total / len(dataloader_train)
            self.setPrint('[{}] Epoch Training loss: {:.4f}'.format(epoch, loss_train_avg))
            total_loss[epoch - 1] = round(loss_train_avg, 4)

            val_loss, predictions, true_vals = self.evaluate(dataloader_test)
            val_f1 = self.f1_score_func(predictions, true_vals)
            total_score[epoch - 1] = round(val_f1, 4)

            self.setPrint('[{}] Validation loss: {:.4f}'.format(epoch, val_loss))
            self.setPrint('[{}] F1 Score : {:.4f}'.format(epoch, val_f1))

        # generate graph
        plt.ion()
        self.generate_graph()
        self.grid_loss_graph(total_loss)
        self.grid_f1_graph(total_score)
        self.init_graph()

    # ##############################################################


if __name__ == "__main__":
    distil_bert = PytorchDistilBERT()
    distil_bert.check_cuda()
    distil_bert.run()