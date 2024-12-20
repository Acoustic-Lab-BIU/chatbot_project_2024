from google.cloud import texttospeech
import os 

def tts(text, lang, voice):
    # Instantiates a client and set the text input to be synthesized
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    # Build the voice request, select the language code ("en-US") and the ssml
    voice = texttospeech.VoiceSelectionParams(language_code=lang, ssml_gender=voice)
    # set type of returned audio to be mp3
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # send the request
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    name = "output" + ".mp3"

    if os.path.exists(name):
        # Delete the old "name.mp3"
        os.remove(name)
        
    with open(name, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
    return name
