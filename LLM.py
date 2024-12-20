import google.generativeai as genai
import API_KEY # Import your own api key from google gemini models.

def llm_query(prompt):
    # config the api key and the requested model
    genai.configure(api_key=API_KEY.API_KEY)

    ''' Options:
        Fast - gemini-1.5-flash-8b
        Balanced - gemini-1.5-flash '''
    model = genai.GenerativeModel("gemini-1.5-flash-8b")

    # send the prompt to the model and print the answer
    response = model.generate_content(prompt)
    return response.text
