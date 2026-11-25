import os, time
from TTS.api import TTS
os.makedirs("out", exist_ok=True)
text = open("tts/text.txt").read().strip()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print([s for s in tts.speakers][:60])
for sp in ["Damien Black", "Viktor Eka", "Craig Gutsy", "Torcull Diarmuid"]:
    t = time.time()
    try:
        tts.tts_to_file(text=text, speaker=sp, language="de", file_path=f"out/xtts_{sp.split()[0].lower()}.wav")
        print(sp, "took", round(time.time() - t), "s")
    except Exception as e:
        print("fail", sp, e)
