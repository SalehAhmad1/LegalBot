from transformers import pipeline
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
        print(f'self.device: {self.device}')
        self.pipe = pipeline("text-generation",
                            model="Equall/Saul-7B-Instruct-v1",
                            device=self.device,
                            # dtype=torch.bfloat16,
                            low_cpu_mem_usage=True)

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
        messages = [
            {"role": "user", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{prompt}"},
            ]

        output = self.pipe(messages, max_length=max_new_tokens, return_full_text=False)
        print('result')
        print(output)
        return output