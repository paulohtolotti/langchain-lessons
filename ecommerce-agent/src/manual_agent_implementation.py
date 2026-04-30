import ollama
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


def get_product_price(product: str) -> float:
    """
    Search for the price of a product in the catalog
    
    Args:
        product: Name of the product
    Returns:
        Full price of the product
    """
    print(f"Searching for {product}", )
    prices = {"laptop": 2500.55, "headphone": 350.55, "keyboard": 60.00}
    return prices.get(product.lower(), 0)


def calculate_final_price_with_discount(price: float, discount_tier: str) -> float:
    """
    Calculate the final price of the product with discount.

    Args:
        price: Full price of the product
        discount_tier: Name of the discount tier. E.g: bronze, silver or gold
    Returns:
        The price of the product with the discount applied. E.g: 100 * (1 - 0.1)
    """
    print(f"Applying discount for {price:.2f}", )
    discounts = {"bronze": 0.05, "silver": 0.15, "gold": 0.30}
    return round(price * (1 - discounts.get(discount_tier, 0)), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_final_price_with_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def chat_tracing(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)



@traceable(name="manual_tool")
def run_agent(question: str) -> None:

    tools_dict = {
        "get_product_price": get_product_price,
        "calculate_final_price_with_discount": calculate_final_price_with_discount
    }
    
    print("=" * 60)
    sys_message ="""
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
        """


    messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": question,}
    ]

    for i in range(0, MAX_ITERATIONS):
        print(f"Iter {i}")

        # Query
        response = chat_tracing(messages=messages)
        ai_message = response.message

        tools_calls = ai_message.tool_calls

        # Quando não chama mais nenhuma tool, o loop acabou
        if not tools_calls:
            print(f"Full ai message {ai_message}")
            print(f"Final message: {ai_message.content}")
            return ai_message.content
        
        tool_call = tools_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool selected {tool_name} with args {tool_args}")
        tool_to_use = tools_dict.get(tool_name)

        
        if tool_to_use is None:
            raise ValueError(f"{tool_name} not found")
        
        # Action
        observation = tool_to_use(**tool_args)

        print(f"Tool result: {observation}")

        # Append das informações do loop atual para manter o contexto
        #
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )

    print("ERROR: Max iterations reached without final answer")
    return None


def main():
    # Prompt que usa as tools
    run_agent("What is the price of the keyboard after applying the bronze discount?")
    
    # Prompt que não precisa de tools
    #run_agent("What is the result of d(3x +2)/dx ?")

if __name__ == "__main__":
    main()