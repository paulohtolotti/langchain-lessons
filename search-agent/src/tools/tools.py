from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

@tool("web_search")
def search(query: str) -> str:
    """
    Tool for web searching.
    Args:
        query: content to query
    Returns:
        content searched
    """
    
    print(f"Static query {query}")
    return tavily.search(query=query)

@tool("monster_status")
def monster_status(monster_name: str) -> str:
    """
    Tool for checking yugioh cards attack and defense.
    Args:
        monster_name:Contains a monster name
    Returns:
        str contaning monster name, attack and defense
    """
    cards = {
        "black_magician": {
            "attack": 2500,
            "defense": 2100
        },
        "black_luster_soldier": {
            "attack": 3000,
            "defense": 2500
        }
    }
    monster = monster_name.lower().replace(" ", "_")

    if monster not in cards:
        return f"Monster {monster_name} not found"
        
    return f"{monster} attack: {cards[monster]["attack"]} def {cards[monster]["defense"]}"
