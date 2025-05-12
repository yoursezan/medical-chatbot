from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# Convert to embeddings
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save to disk
vectorstore.save_local(DB_FAISS_PATH)

print("✅ Vectorstore created successfully at", DB_FAISS_PATH)
