from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langsmith import traceable

load_dotenv()


MAX_ITERATIONS = 5
MODEL = "openai/gpt-oss-120b"

@tool
def get_product_price(product: str) -> float:
    """Search for the price of a product in the catalog"""
    print(f"Searching for {product}", )
    prices = {"laptop": 2500.55, "headphone": 350.55, "keyboard": 60.00}
    return prices.get(product.lower(), 0)

@tool
def calculate_final_price_with_discount(price: float, discount_tier: str) -> float:
    """Use this tool whenever the user asks for a discounted, final, promotional,
    or tier-adjusted price. Do not calculate discounts yourself. The result of
    this tool is the only valid final discounted price.
    """
    print(f"Applying discount for {price:.2f}", )
    discounts = {"bronze": 0.05, "silver": 0.15, "gold": 0.30}
    return round(price * (1 - discounts.get(discount_tier, 0)), 2)


@traceable(name="agent loop")
def run_agent(question: str) -> None:
    tools = [get_product_price, calculate_final_price_with_discount]
    tools_dict = {t.name: t for t in tools} # {get_product, apply_discount}
    
    # Works
    #llm = init_chat_model(model="qwen3:1.7b", model_provider="ollama", temperature=0, )
    llm = init_chat_model(f"groq:{MODEL}", temperature=0.2,)

    # Anexando as tools ao modelo. Modelo precisa ser compatível com tools
    llm_with_tools = llm.bind_tools(tools) 
    print("=" * 60)

    sys_message = SystemMessage(content="""
        You are a shopping assistant.

        Use tools to answer pricing questions.

        Rules:
        1. Product prices must always be obtained with get_product_price.
        2. If the user asks for a discounted/final price and provides a discount tier,
        first call get_product_price, then call apply_discount.
        3. Do not ask the user for product price. Use get_product_price.
        4. Do not ask the user how to calculate discounts. Use apply_discount.
        5. Ask a follow-up question only if the user did not provide the product name
        or did not provide the discount tier when a discount is requested.
        6. Never calculate discounts yourself.
        7. If the product is not found, reply to the user that the requested product is not registered.
        """)


    messages = [
        sys_message,
        HumanMessage(content=question),
    ]
    
    for i in range(0, MAX_ITERATIONS):
        print(f"Iter {i}")

        # Query
        ai_message = llm_with_tools.invoke(messages)

        tools_calls = ai_message.tool_calls

        # Quando não chama mais nenhuma tool, o loop acabou
        if not tools_calls:
            print(f"Final message: {ai_message.content}")
            return ai_message.content
        
        tool_call = tools_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool selected {tool_name} with args {tool_args}")
        tool_to_use = tools_dict.get(tool_name)

        
        if tool_to_use is None:
            raise ValueError(f"{tool_name} not found")
        
        # Action
        observation = tool_to_use.invoke(tool_args)

        print(f"Tool result: {observation}")

        # Append das informações do loop atual para manter o contexto
        #
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without final answer")
    return None


def main():
    run_agent("What is the price of the NASA ROCKET after applying the bronze discount?")


if __name__ == "__main__":
    main()