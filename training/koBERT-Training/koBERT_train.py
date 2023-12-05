import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm

# ★ Hugging Face를 통한 모델 및 토크나이저 Import
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import os

# ★
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

"""# **모델 파라미터 및 기본 환경 구축**"""

# Setting parameters
max_len = 64
batch_size = 192
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 2e-5


'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| max_len = 64 | batch_size = 192 | warmup_ratio = 0.1 | num_epochs = 5 | max_grad_norm = 1 | log_interval = 200 | learning_rate = 2e-5 |
epoch:5 | user(남들 보는 앞에서 교수님한테 엄청 깨졌어.) | Predicted Sentiment(당황이)
epoch:5 | user(하루 종일 일에 쫓기는 기분이야.) | Predicted Sentiment(분노가)
epoch:5 | user(친구랑 싸워서 아직도 분이 안 풀려.) | Predicted Sentiment(분노가)
epoch:5 | user(내 계획은 이게 아니었는데...) | Predicted Sentiment(슬픔이)
epoch:5 | user(이번 발표는 준비를 제대로 못 해서 걱정이다.) | Predicted Sentiment(불안이)
epoch:5 | user(그래도 생각보다는 성적이 괜찮게 나왔어.) | Predicted Sentiment(기쁨이)
epoch:5 | user(헤어진 전 애인이 보고싶다.) | Predicted Sentiment(상처가)
epoch:5 | user(부모님께 드릴 선물 사러 가는 길인데 신나.) | Predicted Sentiment(기쁨이)
epoch:5 | user(애인에게 받은 선물이 너무 맘에 들어.) | Predicted Sentiment(기쁨이)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        #transform = nlp.data.BERTSentenceTransform(
        #    tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTSentenceTransform:
    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab

    def __call__(self, line):
        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #vocab = self._tokenizer.vocab
        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

"""# **감정분류를 위한 데이터셋 불러오기**"""

#감정분류데이터셋 불러오기
import pandas as pd
data = pd.read_excel('/workspace/dataset/감성대화말뭉치.xlsx')

data2 = pd.read_csv('/workspace/dataset/5차년도_2차.csv', encoding='cp949')

data.loc[(data['감정_대분류'] == "상처"), '감정_대분류'] = 0  #상처 => 0
data.loc[(data['감정_대분류'] == "불안"), '감정_대분류'] = 1  #불안 => 1
data.loc[(data['감정_대분류'] == "분노"), '감정_대분류'] = 2  #분노 => 2
data.loc[(data['감정_대분류'] == "슬픔"), '감정_대분류'] = 3  #슬픔 => 3
data.loc[(data['감정_대분류'] == "기쁨"), '감정_대분류'] = 4  #기쁨 => 4
data.loc[(data['감정_대분류'] == "당황"), '감정_대분류'] = 5  #당황 => 5

# print(len(data2))
idx = data2[data2['상황'] == "neutral"].index
data2.drop(idx, inplace=True)
# print(len(data2))

data2.loc[(data2['상황'] == "fear"), '상황'] = 1  #공포 => 1
data2.loc[(data2['상황'] == "angry"), '상황'] = 2  #분노 => 2
data2.loc[(data2['상황'] == "disgust"), '상황'] = 2  #혐오 => 2
data2.loc[(data2['상황'] == "sadness"), '상황'] = 3  #슬픔 => 3
data2.loc[(data2['상황'] == "happiness"), '상황'] = 4  #행복 => 4
data2.loc[(data2['상황'] == "surprise"), '상황'] = 5  #놀람 => 5

data_list = []

for ques, label in zip(data['사람문장1'], data['감정_대분류'])  :
    data = []
    data.append(ques)
    data.append(str(label))

    data_list.append(data)

for ques, label in zip(data2['발화문'], data2['상황'])  :
    data = []
    data.append(ques)
    data.append(str(label))

    data_list.append(data)

data3 = pd.read_excel('/workspace/dataset/감성대화말뭉치_Validation.xlsx')

data3.loc[(data3['감정_대분류'] == "상처"), '감정_대분류'] = 0  #상처 => 0
data3.loc[(data3['감정_대분류'] == "불안"), '감정_대분류'] = 1  #불안 => 1
data3.loc[(data3['감정_대분류'] == "분노"), '감정_대분류'] = 2  #분노 => 2
data3.loc[(data3['감정_대분류'] == "슬픔"), '감정_대분류'] = 3  #슬픔 => 3
data3.loc[(data3['감정_대분류'] == "기쁨"), '감정_대분류'] = 4  #기쁨 => 4
data3.loc[(data3['감정_대분류'] == "당황"), '감정_대분류'] = 5  #당황 => 5

data_list_test = []

for ques, label in zip(data3['사람문장1'], data3['감정_대분류'])  :
    data = []
    data.append(ques)
    data.append(str(label))

    data_list_test.append(data)

data_train = BERTDataset(data_list, 0, 1, tokenizer, vocab, max_len, True, False)
data_test = BERTDataset(data_list_test, 0, 1, tokenizer, vocab, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

"""# **모델 학습 및 평가 해보기**"""

#BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


"""# **새로운 문장 테스트 해보기**"""

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("상처가")
            elif np.argmax(logits) == 1:
                test_eval.append("불안이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("기쁨이")
            elif np.argmax(logits) == 5:
                test_eval.append("당황이")
    

            # print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
            return test_eval[0]

test_questions = [
    "남들 보는 앞에서 교수님한테 엄청 깨졌어.",
    "하루 종일 일에 쫓기는 기분이야.",
    "친구랑 싸워서 아직도 분이 안 풀려.",
    "내 계획은 이게 아니었는데...",
    "이번 발표는 준비를 제대로 못 해서 걱정이다.",
    "그래도 생각보다는 성적이 괜찮게 나왔어.",
    "헤어진 전 애인이 보고싶다.",
    "부모님께 드릴 선물 사러 가는 길인데 신나.",
    "애인에게 받은 선물이 너무 맘에 들어."
]

train_history=[]
test_history=[]
loss_history=[]
for e in tqdm(range(num_epochs)):
    train_acc = 0.0
    test_acc = 0.0
    model.train()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        #print(label.shape,out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} | Batch ID {} | Loss {} | Train Accuracy {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} | Train Accuracy {}".format(e+1, train_acc / (batch_id+1)))
    #train_history.append(train_acc / (batch_id+1))

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} | Test Accuracy {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))
    
    print('+'*100)
    print(f'BatchSize[{BatchSize}] | Epochs[{num_epochs}] | LR[{learning_rate}] | Inference 결과 예시')
    for test_q in test_questions:
        print(f'epoch:{e+1} | user({test_q}) | Predicted Sentiment({predict(test_q)})')
    print('+'*100)

"""# **학습 모델 저장**"""

save_dir = '/workspace/model/saved_models/'

model_save_path = os.path.join(save_dir, f'koBERT_batch{batch_size}_lr{learning_rate}_epoch{num_epochs}.pt')  # 전체 모델 저장


torch.save(model, save_dir + f'model_koBERT_batch{batch_size}_lr{learning_rate}_epoch{num_epochs}.pt')  # 전체 모델 저장

torch.save(model.state_dict(), f'state_koBERT_batch{batch_size}_lr{learning_rate}_epoch{num_epochs}.pt')  # 모델 객체의 state_dict 저장

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, f'all_koBERT_batch{batch_size}_lr{learning_rate}_epoch{num_epochs}.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
