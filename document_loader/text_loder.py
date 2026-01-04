from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="write the summary of the following: {poem}",
    input_variables=["poem"]
)
parser=StrOutputParser()
loader=TextLoader("document_loader/pdf_loder.py")

docs=loader.load()
# print(docs[0].page_content)

chain=prompt | model |parser
result=chain.invoke({"poem":docs[0].page_content})
print(result)