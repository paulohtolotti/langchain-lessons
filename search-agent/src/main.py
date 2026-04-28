from dotenv import load_dotenv # Carrega as variáveis de ambiente no ambiente de execução
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq # Modelo para o agente
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek # Modelo para o deepseek
from tools.tools import search, monster_status
from langchain_tavily import TavilySearch
from models.output import AgentResponse

def main():
    load_dotenv()

    llm = ChatGroq(temperature=0.1, model="llama-3.3-70b-versatile")
    llama_llm = ChatOllama(temperature=0.1, model="qwen3:8b") # muito lento
    llama_llm2 = ChatOllama(temperature=0.1, model="llama3.1:8b") # muito lento
    llama_llm3 = ChatOllama(temperature=0.3, model="llama3.2:1b")
    deep_seek_llm = ChatDeepSeek(temperature=0.3, model="deepseek-v4-flash",)

    tools = [TavilySearch(
        name="tavily_search",
        max_results=3,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False
    )]

    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    result = agent.invoke({"messages": [HumanMessage(content="Procure por 3 vagas de trabalho de desenvolvedor Java Júnior remoto no Brasil. Retorne: título, empresa, link e localização.")]})

    print(result)
    print("\n\n\n")



def manual_tool_calling():
    """Exemplo de como chamar uma tool manualmente e passar seu conteúdo ao modelo"""
    tavily = TavilySearch()
    results = tavily.invoke("vagas Java São Paulo")
    llama_llm3 = ChatOllama(temperature=0.3, model="llama3.2:1b")
    response = llama_llm3.invoke(f"Resuma {results}")


# @tool("query_sql")
# def query_sqlite(sql_query: str) -> str:
#     """Quuery SQLITE database"""
#     return f"Mock data"

if __name__ == "__main__":
    main()
