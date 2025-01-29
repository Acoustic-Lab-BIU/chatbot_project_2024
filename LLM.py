import google.generativeai as genai

def llm_query(prompt,creds):
    # config the api key and the requested model
    genai.configure(credentials=creds)

    ''' Options:
        Fast - gemini-1.5-flash-8b
        Balanced - gemini-1.5-flash '''
    model = genai.GenerativeModel("gemini-1.5-flash-8b")

    # send the prompt to the model and print the answer
    response = model.generate_content(prompt)
    return response.text
