if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM
from transformers import pipeline
import torch

class LLM_HF(LLM):
    def __init__(self) -> None:
        '''
        Constructor for the Legal_LLM class.
        Inputs: 
            - None
        Outputs:
            - None
        '''
        super().__init__()
        self.__initiate_LLM()

    def __initiate_LLM(self) -> None:
        '''
        Function to initiate the LLM.
        Inputs:
            - None
        Outputs:    

        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'self.device: {self.device}')
        self.pipe = pipeline("text-generation",
                            model="Equall/Saul-7B-Instruct-v1",
                            device=self.device,
                            # dtype=torch.bfloat16,
                            low_cpu_mem_usage=True)

    def chat(self, context:str, query:str, max_new_tokens:int=256)  -> str:
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
            {"role": "user", "content": f"{prompt}"},
            ]

        output = self.pipe(messages, max_length=max_new_tokens, return_full_text=False)
        print('Generated Response')
        print(output)
        return output