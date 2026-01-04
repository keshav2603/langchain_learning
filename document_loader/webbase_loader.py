from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
url="https://www.flipkart.com/msi-68-58-cm-27-inch-wqhd-ips-panel-anti-flicker-technology-101-srgb-tilt-adjustable-flat-gaming-monitor-mag-275qf/p/itm170182110df8a?pid=MONH9QS9PDKHZ8PY&lid=LSTMONH9QS9PDKHZ8PYQIVBB0&marketplace=FLIPKART&q=gaming+monitor&store=6bo%2Fg0i%2Funb%2Fpp8&srno=s_1_6&otracker=search&otracker1=search&fm=Search&iid=6a146f24-8883-410b-bc82-da0404ef64b5.MONH9QS9PDKHZ8PY.SEARCH&ppt=sp&ppn=sp&ssid=o5wn2o4o9s0000001767546130009&qH=0f95431207e5b67c"
loader=WebBaseLoader(url)

docs=loader.load()
# print(len(docs))
# print(docs)

prompt=PromptTemplate(
    template="answer this {question} from the given text \n {text}",
    input_variables=["question","text"]
)
parser=StrOutputParser()


chain=prompt | model |parser
result=chain.invoke({"question":"refrest rate of the monitor and will it be good coding?","text":docs[0].page_content})
print(result)