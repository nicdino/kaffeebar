import os, time, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
os.makedirs("out", exist_ok=True)
text = open("tts/text.txt").read().strip()
torch.set_num_threads(os.cpu_count())
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
for name, ins in [
    ("qwen_erzaehler_warm", "Ruhige, warme, tiefe Männerstimme eines Dokumentarfilm-Erzählers, mittleres Alter, entspannt und ohne Eile, freundlich erklärend, natürlicher Sprachfluss."),
    ("qwen_erzaehler_hell", "Sympathische, klare Männerstimme um die dreißig, ruhig und erklärend wie in einem hochwertigen Erklärvideo, natürliche Betonung, entspannt."),
]:
    t = time.time()
    wavs, sr = m.generate_voice_design(text=text, language="German", instruct=ins)
    sf.write(f"out/{name}.wav", wavs[0], sr)
    print(name, "took", round(time.time() - t), "s for", round(len(wavs[0]) / sr, 1), "s audio")
