import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings 



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_naive(query:str, retriever, prompt_template:ChatPromptTemplate, llm):
    """
        Implementação ingênua do rag, com todas as etapas feitas separadamente.
        Gera traces separados de uso.
    """
    docs = retriever.invoke(query)

    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context,question=query)

    return messages

def retrieval_with_chains(query:str, retriever, prompt_template:ChatPromptTemplate, llm): 
    """
        Implementa o RAG usando chains do framework. Todo o trace seguirá uma única chamada de prompt.
        Vantagens:
            - Suporte async
            - Type safety
            - Streaming
    """

    """
        Chain de chamadas.
        Invocamos o retrievar, sua saída vai para o prompt template com a query do usuário, vai para a LLM 
        e a saída de tudo vai para o Parser de String final.
        A saída do retriever precisa entrar no format docs, e a saída desse deve entrar no parâmetro context do prompt template
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == '__main__':
    load_dotenv()
    embeddings = PineconeEmbeddings(model='llama-text-embed-v2')
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2,)

    # Cria um retriever, que busca os K vizinhos mais próximos do prompt
    vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    # Retrieval augmented. O context é a augmentation do RAG
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the following question based on the context.

        {context}

        Question: {question}

        Provide a detailed answer"""
    )

    # Raw invocation
    user_question = "What is Pinecone in machine learning ?"

    result_raw = llm.invoke([HumanMessage(content=user_question)])

    print(f"Raw response: {result_raw.content}")

    # Naive RAG pipeline
    print("="*60)
    print("Naive RAG Impl")
    print("="*60)
    messages = retrieval_naive(user_question, retriever=retriever,prompt_template=prompt_template, llm=llm)
    response = llm.invoke(messages)

    print(response.content)
    print("="*60)
    print("RAG WITH LCEL")
    print("="*60)
    lcel_chain = retrieval_with_chains(user_question, retriever=retriever,prompt_template=prompt_template,llm=llm)
    lcel_res = lcel_chain.invoke({"question": user_question})
    print(lcel_res)