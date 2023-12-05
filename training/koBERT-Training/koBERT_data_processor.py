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


idx = data2[data2['상황'] == "neutral"].index
data2.drop(idx, inplace=True)

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


train_data = pd.DataFrame(data_list)
test_data = pd.DataFrame(data_list_test)

train_data.to_csv('/workspace/dataset/Train_Data.csv', index=False)
train_data.to_excel('/workspace/dataset/Train_Data.xlsx', index=False)
test_data.to_csv('/workspace/dataset/Test_Data.csv', index=False)
test_data.to_excel('/workspace/dataset/Test_Data.xlsx', index=False)