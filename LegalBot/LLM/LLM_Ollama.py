if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM

import ollama

class LLM_Ollama(LLM):
    def __init__(self, model_name='llama3.1') -> None:
        '''
        Constructor for the Legal_LLM class.
        Inputs: 
            - None
        Outputs:
            - None
        '''
        super().__init__()
        self.__initiate_LLM(model_name=model_name)

    def __initiate_LLM(self, model_name) -> None:
        '''
        Function to initiate the LLM.
        Inputs:
            - None
        Outputs:    

        '''
        self.model_name = model_name
        self.pipe = None

    def chat(self, context:str, query:str, max_new_tokens:int=256) -> str:
        '''
        Function to chat with the LLM.
        Inputs:
            - context (str): The context of the document.
            - query (str): The query to be asked.
            - max_new_tokens (int): The maximum number of new tokens to generate.
        Outputs:
            - output (str): The output response of the LLM.
        '''
        prompt = self._format_query(context, query)
        messages = [
            {"role": "user", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{prompt}"},]
        
        stream = ollama.chat(model='llama3.1',
                                messages=messages,
                                stream=True)
        
        for chunk in stream:
            yield chunk['message']['content']