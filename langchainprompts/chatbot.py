from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

chathistory=[SystemMessage(content="you are a helpy assistant")]
while True:
    user_input=input("You: ")
    chathistory.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result=model.invoke(chathistory)
    chathistory.append(AIMessage(content=result.content))
    print("BOT: ",result.content)
    print("total token: ",result.usage_metadata["total_tokens"])
    print("input token: ",result.usage_metadata["input_tokens"])
    print("output token: ",result.usage_metadata["output_tokens"])

# print(chathistory)
# messages=[
#     SystemMessage(content="you are a helpy assistant"),
#     HumanMessage(content="tell me about langchain")
# ]
# result=model.invoke(messages)

# messages.append(AIMessage(content=result.content))

# print(messages)

# chat_template=ChatPromptTemplate([
#     ("system","you are a helpful {domain} expert"),
#     ("human","explain in simple term what is {topic}")
# ])
# prompt=chat_template.invoke({"domain":"cricket","topic":"run out"})
# print(prompt)