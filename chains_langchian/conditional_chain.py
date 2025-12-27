from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

llm = HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation",
    max_new_tokens=50,
    temperature=0,
    do_sample=False,
    return_full_text=False
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="""
You are a sentiment classifier.

Classify the following feedback strictly as either "positive" or "negative".

Feedback:
{text}

Return ONLY valid JSON.
{format_instruction}
""",
    input_variables=["text"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)

clf_chain = prompt1 | model | parser2

branch_chain=RunnableBranch(
    (conditon,chain)
)

result = clf_chain.invoke({"text": "worse phone i use in my life"})
print(result.sentiment)
