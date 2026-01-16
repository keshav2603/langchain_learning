from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text="""This isn’t a deep productivity theory book or a technical mastery book. It’s more of a practical mindset guide — something you can read quickly and apply immediately. Some readers find it very helpful for motivation, others find it a bit general — but it’s worth finishing because it might clarify your approach to work & rest."""

loader=PyPDFLoader("document_loader/dl_curr.pdf")

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
) 

a=splitter.split_documents(docs)
print(a[0].page_content)