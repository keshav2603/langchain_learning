from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
querry="let me about a batman who is famous for his elegant batting style "
# text="heelo my name is keshav"
# document=[
#     "helo my name is keshav",
#     "i wont ot be a billionar",
#     "i will achive it soon"
# ]
# vector=embedding.embed_documents(document)
# print(str(vector))
doc_embedding= embedding.embed_documents(documents)
querry_embed=embedding.embed_query(querry)

scores=cosine_similarity([querry_embed],doc_embedding)[0]
index,score=sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]
print(querry)
print(documents[index])
print(score)
