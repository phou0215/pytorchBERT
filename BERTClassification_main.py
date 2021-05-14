# 2021-04-22 Pytorch transformers sentence classification
# -*- coding: utf-8 -*-

import sys
import os
import torch
import re
import pandas as pd
import numpy as np

from konlpy.tag import Okt
from konlpy.tag import Komoran
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
# from torch.optim import optimizer, Adam
from torch.nn import functional as F


# data set control
class PytorchBERT():

    def __init__(self):
        super(PytorchBERT, self).__init__()
        # main test data
        self.df_data = None

        # input 산툴물 정리
        self.list_memo = None
        self.list_label = None
        self.list_special = None
        self.list_rmstring = None

        # split train and test data
        self.Train_Data_X = None
        self.Train_Data_Y = None
        self.Test_Data_X = None
        self.Test_Data_Y = None
        self.model_root_dir = './save_models'
        self.model_dir = ''
        self.model = None
        self.df_data = None
        self.label_index = None

        self.test_rate = 0.2
        self.batch_size = 2
        self.epoch = 5
        self.learning_rate = 1e-5
        self.file_path = './voc_data.xlsx'
        self.device = torch.device('cpu')
        self.pattern = re.compile("[1-9]{1}[.]{1}")

        self.komoran = Komoran()
        self.twitter = Okt()

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

    def generate_model_directory(self):

        # root dir ath check and generate
        if not os.path.isdir(self.model_root_dir):
            os.makedirs(self.model_root_dir, exist_ok=True)

        # generate models directory
        self.model_dir = '/BERT_TRAINING_MODEL_'+self.getCurrent_time()[2]+'/'
        os.makedirs(self.model_root_dir + self.model_dir, exist_ok=True)

    # 특수문자 제거 함수
    def remove_special(self, text):

        # 전화번호 모두 'tel로 치환'
        text_data = re.sub(r'\d{2,3}[-\.\s]*\d{3,4}[-\.\s]*\d{4}(?!\d)', 'tel', text)
        # 화폐는 'money'로 치환
        text_data = re.sub(r'\d{1,3}[,\.]\d{1,3}[만\천]?\s?[원]|\d{1,5}[만\천]?\s?[원]', 'money', text_data)
        text_data = re.sub(r'일/이/삼/사/오/육/칠/팔/구/십/백][만\천]\s?[원]', 'money', text_data)
        text_data = re.sub(r'(?!-)\d{2,4}[0]{2,4}(?!년)(?!.)|\d{1,3}[,/.]\d{3}', 'money', text_data)
        # web 주소는 'url'로 변경
        text_data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),|]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                           'url', text_data)
        # 그 외의 특수문자는 모두 삭제
        text_data = re.sub(r'[-=+,_#/\?^$@*\"※~&%ㆍ!』\‘|\(\)\[\]\<\>\{\}`><\']', '', text_data)
        # 영문 모두 소문자로 변경
        text_data = text_data.lower()
        # 앞서 list_rmstring 선언된 단어들 모두 제거
        for item in self.list_rmstring:
            text_data = text_data.replace(item, "")

        text_data = text_data.strip()
        return text_data

    #  특수한 형태의 문장을 필요 데이터만 추출 후 버림
    def text_filter(self, list_doc):

        list_return = list_doc
        for idx, doc in enumerate(list_return):
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
            doc = "\n".join([s for s in doc.split('\n') if s])
            list_return[idx] = doc

        return list_return

    # Read excel file and generate dataframe, list.
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

            # 데이터 문자열 전처리
            self.df_data['메모'] = self.df_data['메모'].apply(self.remove_special)
            self.df_data['메모분류'] = self.df_data['메모분류'].apply(lambda x: x.strip())

            # print 데이터 분포
            self.setPrint('데이터 분포: \n{}'.format(self.df_data['메모분류'].value_counts()))

            # label change to index format
            self.label_index = self.set_label_index(self.df_data, col_name='메모분류')
            self.df_data['label'] = self.df_data['메모분류'].replace(self.label_index)
            self.setPrint('index of labels: \n{}'.format(self.label_index))

            # data frame index 번호 reset
            self.df_data.reset_index(drop=True)
            df_config1.reset_index(drop=True)
            df_config2.reset_index(drop=True)

            self.list_memo = self.text_filter(self.df_data['메모'].tolist())
            self.list_label = self.df_data['label'].tolist()

            self.Train_Data_X, self.Test_Data_X, \
            self.Train_Data_Y, self.Test_Data_Y = train_test_split(self.list_memo,
                                                                   self.list_label,
                                                                   test_size=self.test_rate,
                                                                   random_state=42,
                                                                   shuffle=True,
                                                                   stratify=self.list_label)

            self.setPrint('Train_Data: {:.2f}%, Test_Data: {:.2f}% '.
                          format((1 - self.test_rate) * 100, self.test_rate * 100))
        except:
            self.setPrint('Error: {}. {}, line: {}'.format(sys.exc_info()[0],
                                                           sys.exc_info()[1],
                                                           sys.exc_info()[2].tb_lineno))

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

    def run(self):

        # raw data polling and pretreatment datas
        self.generate_data()
        # generate save model directory
        self.generate_model_directory()

        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased'", do_lower_case=True, local_files_only=True)
        encoded_data_train = tokenizer.batch_encode_plus(
            self.Train_Data_X,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            max_length=256,
            return_tensors='pt',
            truncation=True
        )

        encoded_data_val = tokenizer.batch_encode_plus(
            self.Test_Data_X,
            add_special_tokens=True,
            return_attention_mask=True,
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

        self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
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

        # Training start
        for epoch in range(1, self.epoch + 1):

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
                    self.setPrint('{}/{} training_loss : {:.4f}'.format(idx, len(dataloader_train),
                                                                        loss.item() / len(batch)))

            torch.save(self.model.state_dict(),
                       self.model_root_dir + self.model_dir + 'finetuned_BERT_epoch_{}.model'.format(epoch))
            self.setPrint('data_volume/finetuned_BERT_epoch_{}.model'.format(epoch))
            self.setPrint('\nEpoch {}'.format(epoch))

            loss_train_avg = loss_train_total / len(dataloader_train)
            self.setPrint('Training loss: {:.4f}'.format(loss_train_avg))

            val_loss, predictions, true_vals = self.evaluate(dataloader_test)
            val_f1 = self.f1_score_func(predictions, true_vals)
            self.setPrint('Validation loss: {}'.format(val_loss))
            self.setPrint('F1 Score : {}'.format(val_f1))


if __name__ == "__main__":
    bert = PytorchBERT()
    bert.check_cuda()
    bert.run()
