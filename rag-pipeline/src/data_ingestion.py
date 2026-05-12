"""
    Módulo que faz o fluxo completo de embedding de um documento de texto.
    1. Carregar o arquivo com texto
    2. Converter para objetos Document do langchain
    3. Chunking do Document
    4. Embedding
    5. Inserção no banco vetorial
"""
import os 
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    
    # Carregando o texto e os documentos
    print("1. Loading")
    path = 'blog.txt'
    loader = TextLoader(path, encoding='utf-8') 
    document = loader.load()

    print("2. Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"Chunks are {type(chunks)}. Created {len(chunks)} chunks")

    print("3. Embeddings")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = PineconeEmbeddings(model="llama-text-embed-v2")
    # Fazendo o embedding manualmente
    # vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    # print(f"Created {len(vectors)} vectors")
    # for vector in vectors:
    #     print(vector)

    # Faz o embedding automaticamente antes de armazenar
    PineconeVectorStore.from_documents(chunks, embedding=embeddings, index_name=os.environ['INDEX_NAME'])

    print("Finished indexing vectors")