# command to run : CUDA_VISIBLE_DEVICES=0 nohup python koGPT2_v4_finetuning_all.py > nohup_finetuning_all_v4.out 2>&1 &

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
from tqdm import tqdm
import os

Q_TKN = "<usr>"  # User Input
A_TKN = "<sys>"  # GPT Output
BOS = '</s>'  # Beginning of Sequence
EOS = '</s>'  # End of Sequence
MASK = '<unused0>'  # placeholder token used for masking during training
SENT = '<unused1>'  # token for separating different sentences in the input
PAD = '<pad>'  # token for padding sequences to a common length

torch.cuda.set_device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device : {device}')
print(f'Count of using GPUs : {torch.cuda.device_count()}')
print(f'Current cuda device : {torch.cuda.current_device()}')

save_dir = f'/workspace/model/saved_v4_models_all'
batch_sizes = [96]
lr_factors = [3]
max_epochs = 10
max_len = 60

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', device_map='auto')

Chatbot_Data = pd.read_csv('/workspace/dataset/chatbot_dataset.csv')
Sentiment_Chatbot_Data = pd.read_csv("/workspace/dataset/sentiment_chatbot_data.csv")
Subsentiment_Chatbot_Data = pd.read_csv("/workspace/dataset/subsentiment_chatbot_data.csv")
Reduced_Subsentiment_Chatbot_Data = pd.read_csv("/workspace/dataset/reduced_subsentiment_chatbot_data.csv")
Merged_Chatbot_Data = pd.read_csv('/workspace/dataset/merged_chatbot_data.csv')
Reduced_Merged_Chatbot_Data = pd.read_csv('/workspace/dataset/reduced_merged_chatbot_data.csv')

# 챗봇 데이터를 처리하는 클래스 (inheriting from 'torch.utils.data.Dataset')
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=60):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats  # Stores the chat data in the instance variable _data
        self.max_len = max_len  # Sets the maximum length of the sequences to max_len
        self.q_token = Q_TKN  # Sets the question token to the constant Q_TKN
        self.a_token = A_TKN  # Sets the answer token to the constant A_TKN
        self.sent_token = SENT  # Sets the sentence token to the constant SENT
        self.eos = EOS  # Sets the end-of-sequence token to the constant EOS
        self.mask = MASK  # Sets the mask token to the constant MASK
        self.tokenizer = koGPT2_TOKENIZER  # Sets the tokenizer to the pre-trained KoGPT-2 tokenizer

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)  # Returns the number of items in the dataset, which is the length of the _data variable

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]  # Retrieves the chat turn (question and answer) at the specified index idx.
        q = turn["Q"]  # 질문을 가져온다.
        q = re.sub(r"([?.!,])", r" ", q)  # 구둣점들을 제거한다.

        a = turn["A"]  # 답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a)  # 구둣점들을 제거한다.

        # Modification: Include Sentiment Label
        sentiment_label = turn["label"]

        # Modification: Append sentiment label to the question tokenization
        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token + str(sentiment_label))
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_toked[1:]  # Constructs the labels for the model, where the question tokens are masked (self.mask) and the answer tokens follow.

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        # mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)  #Creates a binary mask (mask) to indicate which tokens are part of the question (0) and which are part of the answer (1).
        # Modified: 질문(감정태그 포함)과 답변 모두 1로 설정해 학습에 모두 활용
        mask = [1] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)  #Creates a binary mask (mask) to indicate which tokens are part of the question (0) and which are part of the answer (1).
        
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)

        # Generate attention mask
        attention_mask = [1] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)

        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids, attention_mask)


def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    attention_mask = [item[3] for item in batch]

    return (
        torch.LongTensor(np.array(data)).to(device), 
        torch.LongTensor(np.array(mask)).to(device), 
        torch.LongTensor(np.array(label)).to(device),
        torch.LongTensor(np.array(attention_mask)).to(device),
    )

all_train_set = {
    # 'nosentiment': ChatbotDataset(Chatbot_Data, max_len=max_len), 
    # 'sentiment' : ChatbotDataset(Sentiment_Chatbot_Data, max_len=max_len), 
    # 'subsentiment' : ChatbotDataset(Subsentiment_Chatbot_Data, max_len=max_len),
    # 'reduced_subsentiment' : ChatbotDataset(Reduced_Subsentiment_Chatbot_Data, max_len=max_len)
    'merged' : ChatbotDataset(Merged_Chatbot_Data, max_len=max_len),
    # 'reduced_merged' : ChatbotDataset(Reduced_Merged_Chatbot_Data, max_len=max_len)
}


for key in all_train_set:  # key -> 'nosentiment', 'sentiment', 'subsentiment'
    train_set = all_train_set[key]
    for batch_size in batch_sizes:
        for lr_factor in lr_factors:
            #윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
            train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_batch,)
            learning_rate = 1e-5 * lr_factor
            
            # print(f'model: {model}')
            model = model.to(device)
            model.train()
            print(f'model moved to {device}')

            criterion = torch.nn.CrossEntropyLoss(reduction="none")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            Sneg = -1e18

            print('+'*100)
            print(f'{key} model training -> start')
            for epoch in tqdm(range(1, max_epochs+1, 1)):
                model_save_path = os.path.join(save_dir, f'koGPT2_{key}_model_batch{batch_size}_lr{lr_factor}_epoch{epoch}.pth')
                
                # Modified: display performance score
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for batch_idx, samples in enumerate(train_dataloader):  # use when using nohup (only displays progress for whole epochs)
                    optimizer.zero_grad()
                    # Modified: Unpacked samples to include the attention mask
                    token_ids, mask, label, attention_mask = samples
                    # Modified: Pass attention_mask to the model during training
                    out = model(token_ids, attention_mask=attention_mask)
                    out = out.logits      #Returns a new tensor with the logit of the elements of input
                    mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
                    mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
                    loss = criterion(mask_out.transpose(2, 1), label)
                    # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
                    avg_loss = loss.sum() / mask.sum()
                    avg_loss.backward()
                    # 학습 끝
                    optimizer.step()

                    # Modified: Track Accuracy during Training
                    predictions = torch.argmax(mask_out, dim=-1)
                    correct_predictions += torch.sum(predictions == label).item()
                    total_samples += mask.sum().item()
                    total_loss += avg_loss.item()
                
                # Modified: Calculate Accuracy
                accuracy = correct_predictions/total_samples

                # Save the model
                torch.save(model.state_dict(), model_save_path)

                # Modified: Print Epoch Summary with Accuracy Score
                print(f'|{key}|batch{batch_size}|lr{lr_factor}|epoch{epoch}| Loss: {total_loss / (batch_idx + 1):.4f} | Accuracy: {accuracy * 100:.2f}% | Saved')

                # 각 에포크마다 예시 문장에 대한 답변 생성하도록 코드 추가
                q = "이번 발표는 준비를 제대로 못 해서 걱정이다."
                q = re.sub(r"([?.!,])", r" ", q)
                if key == 'nosentiment':
                    sentiment_label = 0
                elif key == 'sentiment':
                    sentiment_label = '불안'
                elif key == 'subsentiment':
                    sentiment_label = '걱정스러운'
                elif key == 'reduced_subsentiment':
                    sentiment_label = '불안한'
                else:
                    sentiment_label = 'X'
                a = ""
                a = re.sub(r"([?.!,])", r" ", a)
                
                while 1:
                    # Modified: fixed the code to use proper corresponding sentiment label, instead of default value of '0'
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + str(sentiment_label) + A_TKN + a)).unsqueeze(dim=0).to(device)

                    q_len = len(koGPT2_TOKENIZER.tokenize(Q_TKN + q + SENT + str(sentiment_label)))
                    a_len = len(koGPT2_TOKENIZER.tokenize(A_TKN + a))

                    # attention_mask = torch.ones_like(input_ids)  # 이건 근데 무의미해지잖아... 죄다 1인데 어텐션 마스크 쓰는 의미가 퇴색
                    attention_mask = torch.zeros_like(input_ids)
                    attention_mask[:, :q_len + a_len] = 1

                    pred = model(input_ids, attention_mask=attention_mask)

                    pred = pred.logits
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print(f'epoch{epoch} test | user({q}) | Chatbot({a.strip()})')
                print('-'*100)
            print(f'{key} model training -> finished')
            print('+'*100)
