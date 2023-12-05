# Friday App

## Docker Environment
- Need to set Port Configuration as **'''-p 8501:8501 -p 8502:8502'''** during Docker Run Command
- All the Requirements(libraries) for running app_st.py(streamlit app for Friday) freezed in **requirements.txt**
  - in order to install and use kobert-tokenizer from skt/koBERT-base-1, we had to run *'''[pip install git+https://git@github.com/SKTBrain/KoBERT.git@master'''** separately.
- Tested in Docker Container environment running on a GPU server with 2 RTX3090 GPUs, while connected through SSH.

## koGPT2
- Fine-Tuned the [koGPT2 v2 Foundation Model](https://huggingface.co/skt/kogpt2-base-v2) built by SKT and shared on HuggingFace
- used the Fine-Tuned koGPT2 model for inference. Chosen model was Fine-Tuned with Hyper Parameters listed below.
  - batch size : 64
  - learning rate : 3e-05
  - epoch : 5
  - num_of_emotions : 15 (our own metric)
- requires GPU for optimal inference speed (Testing was done using 1 RTX3090 GPU)
- final codes used for inference [koGPT.py]

## koBERT
- Fine-Tuned the [koBERT Foundation Model](https://huggingface.co/skt/kobert-base-v1) built by SKT and shared on HuggingFace
- used the Fine-Tuned koGPT2 model for inference. Chosen model was Fine-Tuned with Hyper Parameters listed below.
  - batch size : 64
  - learning rate : 3e-05
  - epoch : 10
  - num_of_emotions : 15 (our own metric)
- requires GPU for optimal inference speed (Testing was done using 2 RTX3090 GPU, but 1 is sufficient)
- --final codes used for inference [koBERT.py]-- -> **due to errors, we decided to embed the codes for running koBERT during inference directly into the Streamlit application codes(app_st.py) for now.**

## Streamlit
- Provides interactive chat environment for the fine-tuned GPT and BERT models to be utilized.
- Incorporates codes for conducting cosine-based music recommendation (def recommend_song)
