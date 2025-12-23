from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)
# class Review(TypedDict):
#     summary: Annotated[str,"give a 5 word summary of the review we get from clint"]
#     sentiment: Literal["pos","neg"]
#     pros:Annotated[Optional[list[str]],"list down the prows mentioned about the product"]
class Review(BaseModel):
    summary: str=Field(description="summary of the review")
    sentiment: Literal["pos","neg"]
    pros:Annotated[Optional[list[str]],"list down the prows mentioned about the product"]



model=ChatHuggingFace(llm=llm)

structured_model=model.with_structured_output(Review)
result=structured_model.invoke("the hardware is good but the software is a bit lagy the best thing about it is the build quality if it fall it will break the ground it will not impact and ddesign is also good")
print(result)
