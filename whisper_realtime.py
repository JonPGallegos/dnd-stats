import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Set up model: "base", "small", "medium", "large-v2"
model_size = "small"
model = WhisperModel(model_size, compute_type="int8", device='cpu')  # Use "int8" or "float32" if needed

samplerate = 16000
blocksize = 4000
audio_queue = queue.Queue()

# Callback to collect audio blocks
def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# Start audio stream
stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=blocksize)
stream.start()
rec_int = False
print("Listening with Whisper... (Ctrl+C to stop)")
try:
    buffer = np.empty((0,), dtype=np.float32)

    while True:
        block = audio_queue.get()
        block = block.flatten()
        buffer = np.concatenate((buffer, block))
        seconds = 5
        # Run recognition every ~5 seconds
        if len(buffer) >= samplerate * seconds:
            segment = buffer[:samplerate * seconds]
            buffer = buffer[samplerate * seconds:]

            segments, _ = model.transcribe(segment, language="en")
            segments = list(segments)
            if segments:
                txt = segments[-1].text.strip()
                # print(txt)
                if 'slave' in txt.lower():
                    txt_record = txt
                    rec_int = True
                    print('Slave Recognized\n')
                elif rec_int and 'work' not in txt.lower():
                    txt_record += ' ' + txt
                    print('Slave Still Recognized\n')
                elif rec_int and 'work' in txt.lower():
                    print('\nWork Recognized')
                    txt_record += ' ' + txt

                    txt_start = txt_record.lower().rfind('slave')
                    txt_end = txt_record.lower().find('work')

                    print(txt_record[txt_start+6:txt_end])
                    rec_int = False
except KeyboardInterrupt:
    print("\nExiting...")
    stream.stop()
