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


prompt=PromptTemplate(
    template="generate 5 intresting fact about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({"topic":"criket"})

print(result)
chain.get_graph().print_ascii()