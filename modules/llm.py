# llm_wrapper.py
from openai import AsyncOpenAI
import os
import asyncio
import re
import json

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

    def _extract_unsupported_params(self, error_message):
        """
        Extract unsupported parameter names from error messages
        Returns a list of parameter names that are not supported
        """
        print(f"Extracting from error: {error_message}")
        
        try:
            if '{' in error_message:
                json_part = error_message[error_message.find('{'):error_message.rfind('}')+1]
                # print(f"JSON part: {json_part}")
                
                # 使用ast.literal_eval来解析类似Python字典的字符串
                import ast
                try:
                    error_data = ast.literal_eval(json_part)
                    # print(f"Parsed error data with ast: {error_data}")
                    
                    if 'error' in error_data and 'param' in error_data['error']:
                        param = error_data['error']['param']
                        print(f"Found unsupported param: {param}")
                        return [param]
                except Exception as ast_error:
                    print(f"AST parsing failed: {ast_error}")
                    
        except Exception as e:
            print(f"Failed to extract unsupported params from JSON: {e}")
        
        # 备用方法：从特定模式中提取参数名
        try:
            # 匹配 "Unsupported value: 'parameter_name'" 模式
            match = re.search(r"Unsupported value: '([^']+)'", error_message)
            if match:
                param = match.group(1)
                print(f"Found unsupported param from pattern: {param}")
                return [param]
            
            # 匹配 "'param': 'parameter_name'" 模式
            match = re.search(r"'param': '([^']+)'", error_message)
            if match:
                param = match.group(1)
                print(f"Found unsupported param from param field: {param}")
                return [param]
                
        except Exception as e:
            print(f"Failed to extract quoted params: {e}")
        
        print("No unsupported params found")
        return []

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
        # Build params dict, exclude None values
        if "gemini-2.0-flash" in self.model_id.lower():
            reasoning_effort = None  # Not supported (sorry for ugly hardcoding)
        params = {'model': self.model_id, 'messages': messages}
        if temperature is not None:
            params['temperature'] = temperature
        if reasoning_effort is not None:
            params['reasoning_effort'] = reasoning_effort
        if response_format is not None:
            params['response_format'] = response_format
        
        # Add default params
        params.update({'top_p': 1.0, 'n': 1, 'stream': False})
        
        try:
            completion = await self.client.chat.completions.create(**params)
            return completion.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"Error calling API: {error_msg}")
            
            # Retry without unsupported params
            if 'unsupported' in error_msg.lower() or 'not supported' in error_msg.lower():
                unsupported = self._extract_unsupported_params(error_msg)
                if unsupported:
                    print(f"Retrying without: {unsupported}")
                    for param in unsupported:
                        params.pop(param, None)
                    
                    completion = await self.client.chat.completions.create(**params)
                    return completion.choices[0].message.content
            else:
                print("Not a parameter issue, re-raising exception.")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test LLM API call")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model ID to use")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for LLM calls in seconds")
    args = parser.parse_args()
    # Example usage with API
    try:
        # Make sure to set your API key as environment variable:
        # export OPENROUTER_API_KEY="your_api_key_here"
        llm = LLM(model_id=args.model, timeout=args.timeout)
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Explain how AI works in simple terms."},
        ]
        outputs = asyncio.run(llm.chat(messages))
        print("API Response:", outputs)
    except Exception as e:
        print(f"API call failed: {e}")
