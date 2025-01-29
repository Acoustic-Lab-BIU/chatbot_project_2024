import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Host APIs:")
    for i in range(p.get_host_api_count()):
        print(f"  {i}: {p.get_host_api_info_by_index(i)['name']}")
    
    print("\nInput Devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"  {i}: {dev['name']} (Host API: {dev['hostApi']})")
    
    print("\nOutput Devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxOutputChannels'] > 0:
            print(f"  {i}: {dev['name']} (Host API: {dev['hostApi']})")
    
    p.terminate()

# Call the function to list devices
list_audio_devices()