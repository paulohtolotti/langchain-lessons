import os

from dotenv import load_dotenv
from nodes.node import xpto_fun

load_dotenv()

if __name__ == "__main__":
    print(xpto_fun())
    print(os.environ.get('GROQ_API_KEY'))