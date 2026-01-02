from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
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

prompt2=PromptTemplate(
    template="explan this {joke}",
    input_variables=["joke"]
)

parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chain.invoke({"topic":"ai"})


print(result)