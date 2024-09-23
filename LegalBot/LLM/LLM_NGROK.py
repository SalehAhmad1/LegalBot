if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM

import requests

class LLM_NGROK(LLM):
    def __init__(self, ngrok_url: str) -> None:
        '''
        Constructor for the LLM_NGROK class.
        Inputs: 
            - ngrok_url (str): The ngrok URL where the LLM is hosted.
        Outputs:
            - None
        '''
        super().__init__()
        self.ngrok_url = ngrok_url

    def __initiate_LLM(self) -> None:
        '''
        Function to initiate the LLM.
        This function does not need additional setup as the LLM is hosted remotely.
        Inputs:
            - None
        Outputs:    
            - None
        '''
        pass

    def chat(self, context: str, query: str, max_new_tokens: int = 256) -> str:
        '''
        Function to chat with the LLM hosted on ngrok.
        Inputs:
            - context (str): The context of the document.
            - query (str): The query to be asked.
            - max_new_tokens (int): The maximum number of new tokens to generate.
        Outputs:
            - output (str): The generated response from the LLM.
        '''
        prompt = self._format_query(context, query)
        
        data = {
            "text": prompt,
            "max_new_tokens": max_new_tokens
        }

        response = requests.post(f"{self.ngrok_url}/generate", json=data)

        if response.status_code == 200:
            return response.json().get("generated_text", "")
        else:
            raise Exception(f"Error in LLM request: {response.status_code} - {response.text}")
