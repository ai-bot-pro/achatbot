import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(info)
    if info["maxInputChannels"] > 0:
        print(f"Device input index {info['index']} - {info['name']}")
    if info["maxOutputChannels"] > 0:
        print(f"Device output index {info['index']} - {info['name']}")
