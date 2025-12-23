from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template=ChatPromptTemplate([
    ("system","you are a helpful customer support aggent"),
    MessagesPlaceholder(variable_name="chat_history")
    ("human","{querry}")
])