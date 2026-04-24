from langchain_core.messages.ai import (AIMessage,)

class Analyzer:

    @staticmethod
    def show_metrics(response: AIMessage) -> None:
        metrics = dict()
        metadata = response.response_metadata

        metrics['model'] = metadata['model_name']

        # Modelos locais do Ollama não possuem métricas de consumo de token
        if metrics['model'] == 'gemma3:1b':
            return
        
        metrics["total_tokens"] = metadata['token_usage']['total_tokens']
        metrics["prompt_tokens"] = metadata['token_usage']['prompt_tokens']
        metrics["model_tokens"] = metadata['token_usage']['completion_tokens']
        metrics['time'] = f"{round(metadata['token_usage']['completion_time'], 3)}s"
        

        print(f"""\t\t======METRICS ======
        Total tokens used: {metrics["total_tokens"]}
        Input tokens: {metrics["prompt_tokens"]}
        Model tokens: {metrics["model_tokens"]}
        Time used: {metrics['time']}
        Model used: {metrics['model']}
        """)
