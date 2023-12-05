# koGPT2 Training

## Docker Environment
- Need to set Port Configuration as **[-p 8501:8501]** during Docker Run Command
- All the Requirements(libraries) for running app_st.py(streamlit app for Friday) freezed in **requirements.txt**
  - in order to install and use kobert-tokenizer from skt/koBERT-base-1, we had to run **[pip install git+https://git@github.com/SKTBrain/KoBERT.git@master]** separately.
- Tested in Docker Container environment running on a GPU server with 2 RTX3090 GPUs, while connected through SSH.

## Dataset
- custom built Dataset by concatenating [Songys - Chatbot Data](https://github.com/songys/Chatbot_data) & [AI Hub - 감성대화말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)

## Foundation Model
- Fine-Tuned [SKT-AI - koGPT2v2](https://github.com/SKT-AI/KoGPT2) Pre-Trained Model with various batchsize/lr/epoch settings.
  - batch size : 32, 64, 96, 128
  - learning rate : 1e-05, 2e-05, 3e-05, 4e-05, 5e-05
  - epoch : 1~20
  - num_of_emotions : 3, 6, 15(our chosen metric)
- requires GPU for optimal inference speed (Fine-Tuning was done using 2 RTX3090 GPUs each running different fine-tuning instances.)

## Fine-Tuning
- Provided **Input Sentence + Emotion Label** and Fine-Tuned the model to generate appropriate Response sentence.
- run by running bash command: **[CUDA_INVISIBLE_DEVICES=0 nohup python koGPT2_v4_finetuning_all.py &]**
  - to test the results, run by running bash command: **[CUDA_INVISIBLE_DEVICES=0 python koGPT2_v4_run_all.py]**
