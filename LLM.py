import google.generativeai as genai

# def llm_query(prompt,creds):
#     # config the api key and the requested model
#     genai.configure(credentials=creds)

#     ''' Options:
#         Fast - gemini-1.5-flash-8b
#         Balanced - gemini-1.5-flash '''
#     model = genai.GenerativeModel("gemini-1.5-flash-8b")

#     # send the prompt to the model and print the answer
#     response = model.generate_content(prompt)
#     return response.text

class llm_query:
    def __init__(self,creds,sys_prompt=''):
        genai.configure(credentials=creds)
        generation_config = genai.GenerationConfig(
                                                    temperature=0.7,        # Adjust this value (e.g., 0.1 for very deterministic or 1.0 for more creative output)
                                                    max_output_tokens=200,  # Maximum length of the generated output
                                                    top_k=39,               # Limits token selection to the top 50 options
                                                    top_p=0.9               # Uses nucleus sampling (selects from the top 90% cumulative probability)
                                                )
        self.model = genai.GenerativeModel("gemini-1.5-flash-8b",system_instruction=sys_prompt,generation_config=generation_config)

    def __call__(self,prompt):
        response= self.model.generate_content(prompt)
        return response.text