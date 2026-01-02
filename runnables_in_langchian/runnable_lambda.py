from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain1=prompt1|model|parser

def wordcounter(text):
    return len(text.split())


chain=RunnableParallel({
    "joke": RunnablePassthrough(),
    "exp":  RunnableLambda(wordcounter)
})

final_chain=chain1|chain

result=final_chain.invoke({"topic":"cricket"})


print(result)