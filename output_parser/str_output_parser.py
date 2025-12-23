from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


template1=PromptTemplate(
    template="write a detailed report on the {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="write a  5 line summary on the following test. /n {text}",
    input_variables=["text"]
)

# prompt1=template1.invoke({"topic":"black hole"})

# result=model.invoke(prompt1)

# prompt2=template2.invoke({"text":result.content})


# result2=model.invoke(prompt2)
# print(result2.content)


parser=StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":"blackhole"})
print(result)