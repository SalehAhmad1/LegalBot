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
            self.system_prompt = '''WHo are you? You are a legal expert chatbot.
            What is your job? When given a legal document/piece of text and a question as a prompt, your job is to use that context, and generate a relavant response. 
            What to do if the information is less? If the information is less than what is required to give an elaborate response, simply state at the beginning that the information provided is not enough for the information to be accurate and then give a  generic answer to the question. 
            What should be the tone of your outputs? Keep the answer, questions and provide information in a clear, confident, and conversational manner. Avoid using phrases that imply uncertainty or limitations. Respond directly as if you are providing the information yourself.\n'''

    def _format_query(self, context:str, query:str) -> str:
        return f'Content: {context}\n\nPrompt: {query}\n\nAnswer: '.strip()
    
    def chat():
        raise NotImplementedError("This method should be implemented by the subclass")