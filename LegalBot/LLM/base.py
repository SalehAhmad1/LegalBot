class LLM:
    def __init__(self, prompt:str=None) -> None:
        self.initialize_system_prompt(prompt)
        pass

    def _initiate_LLM(self) -> None:
        raise NotImplementedError("This method should be implemented by the subclass")

    def initialize_system_prompt(self, prompt:str=None) -> None:
        if prompt != None:
            self.system_prompt = prompt
        elif prompt == None:
            self.system_prompt = '''You are a legal expert. You are given a legal document and a question. You should answer the question using the document as context. You should not make up an answer.'''

    def _format_query(self, context:str, query:str) -> str:
        return f'Content: {context}\n\nPrompt: {query}\n\nAnswer: '.strip()
    
    def chat():
        raise NotImplementedError("This method should be implemented by the subclass")