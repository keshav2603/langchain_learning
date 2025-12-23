from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

st.header("do research ")
user_input=st.text_input("enter your prompt")

if st.button("ENTER"):
    result=model.invoke(user_input)
    st.write(result.content)
