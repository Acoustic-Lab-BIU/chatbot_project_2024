import os
import queue
import re
import sys
import LLM
from google.cloud import speech
import TTS
from playsound import playsound
import time
import pyaudio
import tkinter as tk
import signal
from threading import Thread
import pygame  
from google.oauth2 import service_account
import wave
from contextlib import contextmanager
import numpy as np
import struct

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
Pause = False

@contextmanager
def ignore_stderr(enable=True):
    if enable:
        devnull = None
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr = os.dup(2)
            sys.stderr.flush()
            os.dup2(devnull, 2)
            try:
                yield
            finally:
                os.dup2(stderr, 2)
                os.close(stderr)
        finally:
            if devnull is not None:
                os.close(devnull)
    else:
        yield

class RedirectedOutput:
    def __init__(self, textbox_widget):
        self.textbox_widget = textbox_widget
        self.current_line = ""
        self.is_speech_line = True
    
    def write(self, text, update_now=True):
        if not text.strip():
            return

        # Handle timing information (enclosed in curly braces)
        if text.strip().startswith('{'):
            self.is_speech_line = False
            self.current_line = ""
            self.textbox_widget.insert(tk.END, "\n" + text.strip() + "\n\n")
        
        # Handle LLM response (starts with newline)
        elif text.startswith('\n'):
            self.is_speech_line = False
            self.current_line = ""
            # Just add the response text without labels
            formatted_text = "\n" + text.strip() + "\n"
            self.textbox_widget.insert(tk.END, formatted_text)
        
        # Handle speech recognition
        elif update_now and self.is_speech_line:
            # Get current text position
            end_index = self.textbox_widget.index("end-1c")
            last_line_start = self.textbox_widget.index(f"{end_index} linestart")
            
            # If there's a current speech line, remove it
            if self.current_line:
                self.textbox_widget.delete(last_line_start, end_index)
            
            # Update with new text, without "User:" prefix
            self.current_line = text.strip()
            if self.current_line:
                self.textbox_widget.insert(last_line_start, self.current_line)
        
        # Handle other output
        else:
            self.is_speech_line = True
            self.current_line = ""
            # Remove any "User:" or "Assistant:" prefixes
            clean_text = text.replace("User: ", "").replace("Assistant: ", "")
            self.textbox_widget.insert(tk.END, clean_text)
        
        # Always scroll to see latest text
        self.textbox_widget.see(tk.END)
        self.textbox_widget.update_idletasks()

    def flush(self):
        pass

class RespeakerAudio(object):
    def __init__(self,mono=True, channels=None, suppress_error=True):
        self.on_audio = self._fill_buffer
        with ignore_stderr(enable=suppress_error):
            self.pyaudio = pyaudio.PyAudio()
        self.available_channels = None
        self.channels = channels
        self.device_index = None
        self.rate = 16000
        self.bitwidth = 2
        self.bitdepth = 16
        self.nb_frames = 1024 # better fits our needs than the original 1024 value
        self.format = pyaudio.paInt16
        self.stream_callback_firstcall=True
        self.tlast = 0
        self.cbcount=0
        self.cbmoy=0
        self.M2=0
        self.mono = mono
        self._buff = queue.Queue()

        # find device
        count = self.pyaudio.get_device_count()
        for i in range(count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info["name"]
            chan = info["maxInputChannels"]
            if name.lower().find("respeaker") >= 0:
                self.available_channels = chan
                self.device_index = i
                break
        if self.device_index is None:
            info = self.pyaudio.get_default_input_device_info()
            self.available_channels = info["maxInputChannels"]
            self.device_index = info["index"]

        if self.available_channels != 6:
            print("You may have to update firmware.")
        if self.channels is None:
            self.channels = range(self.available_channels)
        else:
            self.channels = filter(lambda c: 0 <= c < self.available_channels, self.channels)
        if not self.channels:
            raise RuntimeError('Invalid channels %s. (Available channels are %s)' % (
                self.channels, self.available_channels))


    def __enter__(self: object) -> object:
        if not self.mono:
            self.stream = self.pyaudio.open(
                input=True, start=False,
                format=self.format,
                channels=self.available_channels,
                rate=self.rate,
                frames_per_buffer=self.nb_frames,
                stream_callback=self.stream_callback,
                input_device_index=self.device_index,
            )
        else:
            self.stream = self.pyaudio.open(
            input=True, start=True,
            format=self.format,
            channels=self.available_channels,
            rate=self.rate,
            frames_per_buffer=self.nb_frames,
            stream_callback=self._fill_buffer,
            input_device_index=self.device_index)
            
        self.closed = False
        
        return self
    
    def __exit__(self: object,type: object,value: object,traceback: object,):
        self.stream.stop_stream()
        self.stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self.pyaudio.terminate()


    def __del__(self):
        self.stop()
        try:
            self.stream.close()
        except:
            pass
        finally:
            self.stream = None
        try:
            self.pyaudio.terminate()
        except:
            pass

    def stream_callback(self, in_data, frame_count, time_info, status):
 	# retrieve audio data
        data = np.fromstring(in_data, dtype=np.int16)
        self.on_audio(data)
        return None, pyaudio.paContinue
    
    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        if self.mono:
            data = np.frombuffer(in_data, dtype=np.int16).reshape(self.nb_frames,6)  
            in_data= data[:,1] #''.join(''.join(struct.pack('h', sample) for sample in data[:,1]))
            in_data = in_data.tobytes()
#        print(np.frombuffer(in_data,dtype=np.int16).max())
        self._buff.put(in_data)
        return None, pyaudio.paContinue
    
    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=12,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )
        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(stream: object, responses: object) -> str:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Args:
        responses: List of server responses

    Returns:
        The entire transcribed text.
    """
    prompt = ""
    num_chars_printed = 0
    print(prompt)
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        if not result.is_final:
            # User still talking
            sys.stdout.write(transcript + overwrite_chars + "\r", update_now = True)
            sys.stdout.flush()
            num_chars_printed = len(transcript)

        else:
            # User stopped talking
            print(transcript + overwrite_chars)
            sys.stdout.write(transcript + overwrite_chars, update_now = True)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break
            prompt += transcript      
            num_chars_printed = 0
            break

    return prompt

def update_dict(dict, new_value, size, type, index):
     # In the  five first iteration on the dict
     if dict[type + str(index)] == "":
          dict[type + str(index)] = new_value
     else: 
          # In the other iterations on the dict
          dict[type +'1'] = ""     # delete the oldest question
          for i in range(1,size):
               dict[type + str(i)] = dict[type + str(i+1)]
          dict[type + str(size)] = new_value
     return dict

def main(lang: str, voice: int) -> str:
    """
    Transcribe speech from microphone.
    Return: Transcribed speech

    Steps to set the evniorment variable of the credentials for the STT & TTS:
    1. Create your own google application credentials.
    2. Make sure to enable the STT & TTS services in your google cloud account.
    3. Download the key you created (suppose to be a JSON file)
    4. Set the variable in the next line with the path of your own credentials.
    5. Run the main file and have fun! :)

    ** In the LLM.py file make sure to create an API key for the wanted model,
    and set it in the LLM.py file **
    """
    json_path = 'Google_key.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_path
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    
    # Configurations:
    language_code = lang  # For English: "en-US", For Hebrew: "iw"
    creds = service_account.Credentials.from_service_account_file(json_path)
    client = speech.SpeechClient(credentials=creds)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    questions = {"Q1": "","Q2": "","Q3": "","Q4": "","Q5": "",}   
    answers = {"A1": "","A2": "","A3": "","A4": "","A5": "",}
    free_space_index = 1
    if language_code != "iw":
        great_speech = TTS.tts('Hello. you can start talking in English and I will answer your questions',"en-US",voice)
    else: 
        great_speech = TTS.tts('שלום. אתה יכול להתחיל לדבר בעברית ואני אענה על שאלותיך',"iw",voice)
    play_audio(great_speech)
    # Infinity loop
    while True: 
        # STT
        with RespeakerAudio(mono=True) as stream:
        # with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            responses = client.streaming_recognize(streaming_config, requests)
            transcription = listen_print_loop(stream, responses)
        if len(transcription) == 0:
            continue
        
        questions = update_dict(questions, transcription, 5, "Q", free_space_index)  
        # Prompt enginerring
        prompt = """You are a chatbot that supposed to have a conversation with users at Bar Ilan university in Israel.
                    refer 'the Universty' as Bar Ilan university in Israel . if the question is in Hebrew answer in Hebrew.
                    Dont use asterisks or emojis, try to keep your answer short and to the point.
                    Inject some light humor and wit to keep things fun, but stay professional and avoid excessive formality.
                    Every time I will send you this message and the five latest questions and answers I asked and you replied.
                    I will send you the history of our conversation in two dictionary formats,
                    the first will be the questions, and the second will be the answers accordingly.
                    I want you to return an answer to my latest question everytime based on the previous responses,
                    now this is my question: """ + transcription  + """and
                    this is the history of the questions, and answers: """ + str(questions) + str(answers) #and be as funny as you can.
        start_time_llm = time.time() # Start time (LLM)
        answer = LLM.llm_query(prompt,creds=creds)
        end_time_llm = time.time()
        llm_execution_time = end_time_llm - start_time_llm

        answers = update_dict(answers, answer, 5, "A", free_space_index)
        free_space_index += 1
        if free_space_index > 5:
            free_space_index = 1
        sys.stdout.write(f"\nA: {answer}\n")   # Print LLM response

        # TTS
        start_time_tts = time.time() # Start time (TTS)
        audio_file_name = TTS.tts(answer, language_code, voice)   # TTS request
        end_time_tts = time.time() # Stop time (TTS)
        tts_execution_time = end_time_tts - start_time_tts

        # printing times and playing the answer
        # Format timing information with extra newlines
        print(f"\nLLM execution time: {llm_execution_time:.6f} seconds")
        print(f"TTS execution time: {tts_execution_time:.6f} seconds")
        print(f"\nOverall time: {tts_execution_time + llm_execution_time:.6f} seconds")  
        play_audio(audio_file_name)

def play_audio(file_path):
    """Plays a WAV audio file using PyAudio."""
    # Open the WAV file
    wf = wave.open(file_path, 'rb')
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
        output_device_index=5
    )

    # Read and play audio in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Close stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
    
# This function is activated when the buttons of the voice menu are pressed
def on_voice_select(voice_choice):
    global voice
    voice = voice_choice
    voice_frame.pack_forget()  # Hide voice selection frame
    voice_menu_label.pack_forget()
    language_frame.pack(expand=True)  # Show language selection frame

# This function terminates the program when the STOP button is pressed
def stop_program():
    try:
        # Delete the old "output.mp3"
        if os.path.exists("output.wav"):
            os.remove("output.wav")
        # Stop pygame mixer if it's playing
        pygame.mixer.quit()
    except:
        pass
    
    try:
        # Kill any running threads
        for thread in Thread.enumerate():
            if thread is not Thread.main_thread():
                try:
                    thread._stop()
                except:
                    pass
    except:
        pass
    
    try:
        # Stop the main window
        root.quit()
        root.destroy()
    except:
        pass
    
    # Force exit
    os._exit(0)

# This function handles the English button
def start_function_english():
    headline.pack_forget()
    language_frame.pack_forget()  # Hide language selection frame
    textbox.pack(pady=10)  # Show the textbox
    stop_button.pack(pady=10)  # Show the stop button
    textbox.delete(1.0, tk.END)  # Clear any existing text
    stdout_redirector.write("Start talking in English\n")  # Initial message

    # Start main function in a separate thread
    thread = Thread(target=lambda: main(lang="en-US", voice=voice))
    thread.daemon = True  # Set as daemon so it will be killed when main program exits
    thread.start()

# This function handles the Hebrew button
def start_function_hebrew():
    headline.pack_forget()
    language_frame.pack_forget()  # Hide language selection frame
    textbox.pack(pady=10)  # Show the textbox
    stop_button.pack(pady=10)  # Show the stop button
    textbox.delete(1.0, tk.END)  # Clear any existing text
    stdout_redirector.write("Start talking in Hebrew\n")  # Initial message
     
    # Start main function in a separate thread
    thread = Thread(target=lambda: main(lang="iw", voice=voice))
    thread.daemon = True  # Set as daemon so it will be killed when main program exits
    thread.start()

# This function reveals the textbox where the chat is 
def show_textbox():
    textbox.pack(side=tk.LEFT, pady=10, padx=(10, 0))
    scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=10)

if __name__ == "__main__":

    ###################################
    ############---Main---#############
    ###################################

    # Create the main window
    root = tk.Tk()
    root.title("ChatBot")

    # Set window size and background color
    root.geometry("800x500")
    root.configure(bg='#282c34')

    # Create a frame
    frame = tk.Frame(root, bg='#282c34')
    frame.pack(expand=True)  # Use expand=True -> center the frame

    # Update textbox creation with better styling
    textbox = tk.Text(
        frame,
        height=20,
        width=80,
        bg='#1e1e1e',
        fg='#ffffff',
        font=('Consolas', 12),
        wrap='word',
        padx=10,
        pady=10
    )

    # Add scrollbar
    scrollbar = tk.Scrollbar(frame, command=textbox.yview)
    textbox.configure(yscrollcommand=scrollbar.set)

    # Headline label
    headline = tk.Label(frame, text="Chatbot", bg='#282c34', fg='#ffffff', font=('Arial', 24, 'bold'))
    headline.pack(pady=20)

    # Create the Textbox (initially hidden)
    textbox = tk.Text(frame, height=20, width=80, bg='#f5f5f5', fg='#000000', font=('Arial', 12), wrap='word')

    # Create the redirector instance
    stdout_redirector = RedirectedOutput(textbox)
    sys.stdout = stdout_redirector

    # Create single stop button (initially hidden) - MOVED OUTSIDE FUNCTIONS
    stop_button = tk.Button(frame, text="Stop", command=stop_program, height=3, width=20, 
                        bg='#f44336', fg='#ffffff', font=('Arial', 12, 'bold'))

    # Frames and Labels for the menus
    voice_frame = tk.Frame(frame, bg='#282c34')
    language_frame = tk.Frame(frame, bg='#282c34')
    voice_menu_label = tk.Label(frame, text="Choose Voice", bg='#282c34', fg='#ffffff', 
                            font=('Arial', 18, 'bold'))
    language_menu_label = tk.Label(language_frame, text="Choose Language", bg='#282c34', 
                                fg='#ffffff', font=('Arial', 18, 'bold'))

    # Voice menu buttons
    male_voice_button = tk.Button(voice_frame, text="Male", command=lambda: on_voice_select(1), 
                                height=4, width=25, bg='#4CAF50', fg='#ffffff', 
                                font=('Arial', 12, 'bold'))
    female_voice_button = tk.Button(voice_frame, text="Female", command=lambda: on_voice_select(2), 
                                height=4, width=25, bg='#4CAF50', fg='#ffffff', 
                                font=('Arial', 12, 'bold'))
    neutral_voice_button = tk.Button(voice_frame, text="Neutral", command=lambda: on_voice_select(3), 
                                    height=4, width=25, bg='#4CAF50', fg='#ffffff', 
                                    font=('Arial', 12, 'bold'))

    # Show the voice selection frame initially
    voice_menu_label.pack(pady=10)
    voice_frame.pack(expand=True)
    male_voice_button.pack(pady=10)
    female_voice_button.pack(pady=10)
    neutral_voice_button.pack(pady=10)

    # Language menu buttons
    english_button = tk.Button(language_frame, text="English", command=start_function_english, 
                            height=4, width=25, bg='#4CAF50', fg='#ffffff', 
                            font=('Arial', 12, 'bold'))
    hebrew_button = tk.Button(language_frame, text="Hebrew", command=start_function_hebrew, 
                            height=4, width=25, bg='#4CAF50', fg='#ffffff', 
                            font=('Arial', 12, 'bold'))

    language_menu_label.pack(pady=20)
    english_button.pack(pady=20)
    hebrew_button.pack(pady=20)

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, lambda sig, frame: stop_program())

    # Run the application
    root.mainloop()
