from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("text_data")
DB_DIR = Path("chroma_db")

print("📥 Loading documents...")

documents = []
for txt_file in DATA_DIR.glob("*.txt"):
    loader = TextLoader(str(txt_file), encoding="utf-8")
    documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

print("✂️ Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("🧬 Creating embeddings...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("🗄️ Building vector store...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=str(DB_DIR)
)

db.persist()

print("✅ Knowledge base built successfully!")
