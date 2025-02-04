from google.cloud import texttospeech
import os 
import soundfile as sf
import numpy as np
def tts(text, lang, voice):
    # Instantiates a client and set the text input to be synthesized
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    # Build the voice request, select the language code ("en-US") and the ssml
    voice = texttospeech.VoiceSelectionParams(language_code=lang, ssml_gender=voice)
    # set type of returned audio to be mp3
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,sample_rate_hertz=16000)

    # send the request
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    name = "output" + ".wav"

    if os.path.exists(name):
        # Delete the old "name.mp3"
        os.remove(name)
    with open(name, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
    audio_data, sample_rate = sf.read(name)
    if not np.issubdtype(audio_data.dtype, np.floating):
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    # Normalize the audio to the target peak
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
       normalized_audio = (audio_data / max_val) * 0.05
    else:
        normalized_audio = audio_data  # If audio is silent, no need to normalize

    # Save the normalized audio to a new file
    sf.write(name, normalized_audio, sample_rate)

    return name
