from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from analyzer.analyzer import Analyzer

load_dotenv()

def basic_pipeline():
    # Chain == pipeline architecture
    question = input("What is your question ? ")

    input_template = """
    Answer the following question {question}.
    Output rules
    1. Keep the answer at 6 lines, maximum
    2. Be polite
    3. Answer in the language that the question was asked
    """

    prompt_template = PromptTemplate(
        input_variables=["question"], template=input_template
    )

    model_groq = ChatGroq(temperature=0, model="llama-3.1-8b-instant")
    model_llama = ChatOllama(temperature=0, model="gemma3:1b")

    # Mesma chain de um comando shell e.g: ls -l ~/ | tail -n 5
    chain = prompt_template | model_groq
    response = chain.invoke(input={"question": question})

    return response

def main():

    response = basic_pipeline()
    print(response.content)
    print(response)
    Analyzer.show_metrics(response)


if __name__ == "__main__":
    main()
