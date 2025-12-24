from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
temp1=PromptTemplate(
    template="give me the name , age , city of a frictional character \n {format_info}",
    input_variables=[],
    partial_variables={"format_info":parser.get_format_instructions()}
)

chain = temp1 | model | parser

print(chain.invoke({}))
