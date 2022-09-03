#run sample streamlit script or smtn
#in terminal "streamlit run app.py"
#reload the url will get a change 
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
import gdown
import shutil
from transformers import TextClassificationPipeline
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Fake News Detection")
st.text("Enter an Article and see if it contains fake news or not")
st.text("0 is true news, 1 is fake news")




# gdown.download(
#         "https://drive.google.com/drive/folders/1gVIgMuxdXZt9HhXU2UxuhCfXD6VF9KYu",
#         "/model",
#         quiet=True
#     )


tokenizer = AutoTokenizer.from_pretrained("philbell/roshni_fake_news")

model = AutoModelForSequenceClassification.from_pretrained("philbell/roshni_fake_news")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

article = st.text_input('enter an article') 
if article:
    prediction = pipe(article)
    st.text(prediction)
    st.balloons()


st.text(" ")
st.text(" ")
st.text(" ")

st.text("If you want to learn more about how to train this model,")
st.text("check out the Colab Notebook or Research Paper below")
st.text("https://colab.research.google.com/drive/180kHTXHXV0rllcJ5WDcbbykBXxVqVv2W?usp=sharing")
st.text("https://docs.google.com/document/d/1nwIOhiKW93PXjrQ8pyDstZoEVlp5UA_N74jLDS_jalc/edit?usp=sharing")


