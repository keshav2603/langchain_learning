from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template="generate a tweet about a {topic}",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="generate a linkedin post about a {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()

chain=RunnableParallel({
    "tweet":RunnableSequence(prompt1,model,parser),
    "linkedin":RunnableSequence(prompt2,model,parser)
})

result=chain.invoke({"topic":"ai"})
print(result)