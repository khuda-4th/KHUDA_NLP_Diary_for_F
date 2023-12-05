# koBERT Training

## Dataset
  - [AI Hub - 감성대화말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)

## Foundation Model
- Fine-Tuned the [koBERT Foundation Model](https://huggingface.co/skt/kobert-base-v1) built by SKT and shared on HuggingFace
- used the Fine-Tuned koGPT2 model for inference. Chosen model was Fine-Tuned with Hyper Parameters listed below.
  - batch size : 64
  - learning rate : 3e-05
  - epoch : 10
  - num_of_emotions : 15 (our own metric)
- requires GPU for optimal inference speed (Testing was done using 2 RTX3090 GPU, but 1 is sufficient)
- --final codes used for inference [koBERT.py]-- -> **due to errors, we decided to embed the codes for running koBERT during inference directly into the Streamlit application codes(app_st.py) for now.**

## Fine-Tuning
- A layer for 15 emotion classifications is added on top of the pre-trained BERT base_v1 model, utilizing resources such as Wikipedia for pre-trained.
  - Using a dataset labeled with 15 emotions, the **'''num_classes'''** parameter within the BERTClassifier class is modified to 15.
