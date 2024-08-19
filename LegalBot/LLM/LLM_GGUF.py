if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM
from llama_cpp import Llama
import torch

class LLM_GGUF(LLM):
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
        self.llm = Llama(model_path="../Models/Saul-Instruct-v1.Q8_0.gguf", #model file path
                        n_ctx=3000, #possible (can be increased) context length
                        n_gpu_layers=-1 if self.device == 'cuda' else 0,
                        verbose=False)

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
        output = self.llm(f'''<|im_start|>system
                        {self.system_prompt }<|im_end|>
                        <|im_start|>user
                        {prompt}<|im_end|>
                        <|im_start|>assistant''', 
                        max_tokens=max_new_tokens,  
                        stop=["</s>", "<|im_end|>", "[/INST]"],
                        echo=False,       # Whether to echo the prompt or not
                    )
        return output['choices'][0]['text'].strip()