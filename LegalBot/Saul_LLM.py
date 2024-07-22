from llama_cpp import Llama
import torch

class Legal_LLM:
    def __init__(self):
        '''
        Constructor for the Legal_LLM class.
        Inputs: 
            - None
        Outputs:
            - None
        '''
        self.__initiate_LLM()
        self.system_prompt = '''You are a legal expert. You are given a legal document and a question. You should answer the question using the document. You should not make up an answer. If you donot know the answer simply say, you donot know.'''

    def __initiate_LLM(self):
        '''
        Function to initiate the LLM.
        Inputs:
            - None
        Outputs:    

        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llm = Llama(model_path="../Models/Saul-Instruct-v1.Q8_0.gguf",
                        n_ctx=1000,
                        n_gpu_layers=-1 if self.device == 'cuda' else 0,
                        verbose=False)


    def __format_query(self, context:str, query:str):
        '''
        Function to format the query.
        Inputs:
            - context (str): The context of the document.
            - query (str): The query to be asked.
        Outputs:
            - prompt (str): The formatted prompt.
        '''
        return f'Content: {context}\n\nPrompt: {query}\n\nAnswer: '

    def chat(self, context:str, query:str, max_new_tokens:int=256):
        '''
        Function to chat with the LLM.
        Inputs:
            - context (str): The context of the document.
            - query (str): The query to be asked.
            - max_new_tokens (int): The maximum number of new tokens to generate.
        Outputs:
            - output (str): The output response of the LLM.
        '''
        prompt = self.__format_query(context, query)
        output = self.llm(f'''<|im_start|>system
                        {self.system_prompt }<|im_end|>
                        <|im_start|>user
                        {prompt}<|im_end|>
                        <|im_start|>assistant''', 
                        max_tokens=max_new_tokens,  
                        stop=["</s>", "<|im_end|>"],
                        echo=False,       # Whether to echo the prompt,
                    )
        return output['choices'][0]['text'].strip()