# llm_wrapper.py
from openai import AsyncOpenAI
import os
import asyncio

class ABC:
    def __init__(self):
        pass

    async def chat(self, messages):
        raise NotImplementedError("Subclasses should implement this method")

class LLM(ABC):
    def __init__(self, model_id="gpt-5", timeout=600):
        """
        Args:
            model_id: Model name on OpenRouter
            max_new_tokens: Maximum number of tokens to generate
            api_key: OpenRouter API key (if None, will read from environment variable)
            base_url: OpenRouter API base URL
        """
        # Get API key from environment variable
        from dotenv import load_dotenv
        load_dotenv()  # Load environment variables from .env file
        if 'deepseek' in model_id.lower():
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
        elif 'gemini' in model_id.lower():
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1/"
        
        if api_key is None:
            raise ValueError(f"API key is required. Set API Key for {model_id} environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        self.model_id = model_id

    async def chat(self, messages, temperature=0.7, reasoning_effort="high", response_format=None):
        """
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature (0.0-2.0)
            reasoning_effort: Level of reasoning effort ("low", "medium", "high")
            response_format: Response format specification
        Returns:
            str: Generated response
        """
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                # reasoning_effort=reasoning_effort,
                # response_format=response_format,
                top_p=1.0,  # Add nucleus sampling parameter (default to 1.0)
                n=1,  # Number of completions to generate
                stream=False,  # Disable streaming for simplicity
            )
            # print("Completion received:", completion)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling API: {e}")
            raise

if __name__ == "__main__":
    # Example usage with API
    try:
        # Make sure to set your API key as environment variable:
        # export OPENROUTER_API_KEY="your_api_key_here"
        model_id = "gemini-2.5-flash"
        llm = LLM(model_id=model_id)
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
        ]
        outputs = asyncio.run(llm.chat(messages))
        print("API Response:", outputs)
    except Exception as e:
        print(f"API call failed: {e}")
