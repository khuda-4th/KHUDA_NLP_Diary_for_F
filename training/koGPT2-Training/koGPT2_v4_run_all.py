# command to run : CUDA_VISIBLE_DEVICES=1 nohup python koGPT2_v4_run_all.py > nohup_run_all_v4.out 2>&1 &

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
import os

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

torch.cuda.set_device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device : {device}')
print(f'Count of using GPUs : {torch.cuda.device_count()}')
print(f'Current cuda device : {torch.cuda.current_device()}')

save_dir = f'/workspace/model/saved_v4_models_all'

batch_sizes = [64, 96, 128]
lr_factors = [3]
max_epochs = 10
max_len = 60

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

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

test_questions_sentiment_labels = {
    'nosentiment' : [1, 0, 1, 0, 0, 2, 1, 2, 2],  # 일상다반사 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링
    'sentiment' : ['슬픔', '불안', '분노', '당황', '불안', '기쁨', '상처', '기쁨', '기쁨'],  # [분노, 기쁨, 불안, 당황, 슬픔, 상처]로 레이블링
    'subsentiment' : ['당혹스러운', '스트레스 받는', '분노', '당황', '걱정스러운', '안도', '상처', '기쁨', '만족스러운'],  # [노여워하는, 느긋, 걱정스러운, 당혹스러운, 당황, 마비된, 만족스러운, 배신당한, 버려진, 부끄러운, 분노, 불안, 비통한, 상처, 성가신, 스트레스 받는, 슬픔, 신뢰하는, 신이 난, 실망한, 악의적인, 안달하는, 안도, 억울한, 열등감, 염세적인, 외로운, 우울한, 고립된, 좌절한, 후회되는, 혐오스러운, 한심한, 자신하는, 기쁨, 툴툴대는, 남의 시선을 의식하는, 회의적인, 죄책감의, 혼란스러운, 초조한, 흥분, 충격 받은, 취약한, 편안한, 방어적인, 질투하는, 두려운, 눈물이 나는, 짜증내는, 조심스러운, 낙담한, 환멸을 느끼는, 희생된, 감사하는, 구역질 나는, 가난한, 불우한]으로 레이블링
    'reduced_subsentiment' : ['부끄러운', '스트레스 받는', '분노한', '당황한', '불안한', '편안한', '슬픈', '기쁨', '감사하는'],  # ['분노한', '편안한', '불안한', '당황한', '기쁨', '슬픈', '외로운', '부끄러운', '스트레스 받는', '실망한', '열등감을 느끼는', '회의적인', '후회되는', '자신하는', '감사하는']으로 레이블링
    'merged' : [4, 0, 0, 3, 2, 1, 5, 1, 1],  # [0('분노'), 1('기쁨'), 2('불안'), 3('당황'), 4('슬픔'), 5('상처'), 6('기타')}로 레이블링,
    'reduced_merged' : [7, 8, 0, 3, 2, 1, 5, 4, 14]  # [0:('분노한'), 1:('편안한'), 2:('불안한'), 3:('당황한'), 4:('기쁨'), 5:('슬픈'), 6:('외로운'), 7:('부끄러운'), 8:('스트레스 받는'), 9:('실망한'), 10:('열등감을 느끼는'), 11:('회의적인'), 12:('후회되는'), 13:('자신하는'), 14:('감사하는'), 15:('기타')]로 레이블링
}

all_train_set = [
    # 'nosentiment',
    # 'sentiment', 
    # 'subsentiment',
    # 'reduced_subsentiment',
    'merged',
    'reduced_merged'
]



for key in all_train_set:  # key -> 'nosentiment', 'sentiment', 'subsentiment'
    sentiment_label_list = test_questions_sentiment_labels[key]
    for batch_size in batch_sizes:
        for lr_factor in lr_factors:
            print('+'*100)
            for epoch in range(1, max_epochs+1, 1):
                model_save_path = os.path.join(save_dir, f'koGPT2_{key}_model_batch{batch_size}_lr{lr_factor}_epoch{epoch}.pth')
                try:
                    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', state_dict=torch.load(model_save_path)).to(device)
                except FileNotFoundError:
                    print(f'|{key}|batch{batch_size}|lr{lr_factor}|epoch{epoch}| Model Not Found in the Directory')
                    continue
                print(f'|{key}|batch{batch_size}|lr{lr_factor}|epoch{epoch}| Model loaded')
                
                for q_idx in range(len(test_questions)):
                    q = test_questions[q_idx]  # Modified
                    q = re.sub(r"([?.!,])", r" ", q)
                    # sentiment_label = sentiment_label_list[q_idx]  # Modified
                    a = ""

                    while 1:
                        # Modified: sentiment token을 기본값인 '0' 대신 각 데셋 및 문장에 맞는 값으로 지정
                        '''input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + str(sentiment_label) + A_TKN + a)).unsqueeze(dim=0).to(device)'''
                        input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0).to(device)

                        # Modified: Calculate attention mask dynamically
                        '''q_len = len(koGPT2_TOKENIZER.tokenize(Q_TKN + q + SENT + str(sentiment_label)))'''
                        q_len = len(koGPT2_TOKENIZER.tokenize(Q_TKN + q + SENT))
                        a_len = len(koGPT2_TOKENIZER.tokenize(A_TKN + a))
                        attention_mask = torch.zeros_like(input_ids)
                        attention_mask[:, :q_len + a_len] = 1

                        pred = model(input_ids, attention_mask=attention_mask)
                        # pred = model(input_ids)

                        pred = pred.logits
                        '''gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]'''
                        gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                        if gen == EOS:
                            break
                        a += gen.replace("▁", " ")
                        a = re.sub(r"([?.!,])", r" ", a)
                    '''print(f'epoch{epoch} test | user({q})[{sentiment_label}] | Chatbot({a.strip()})')'''
                    print(f'epoch{epoch} test | user({q}) | Chatbot({a.strip()})')
                print('-'*100)
            print('+'*100)
