import os, sys, json, time, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
shard, n = int(sys.argv[1]), int(sys.argv[2])
os.makedirs("out/vo", exist_ok=True)
items = json.load(open("tts/vo.json"))
mine = [it for k, it in enumerate(items) if k % n == shard]
torch.set_num_threads(os.cpu_count())
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
ref_text = open("tts/text.txt").read().strip()
prompt = m.create_voice_clone_prompt(ref_audio="out/qwen_erzaehler_warm.mp3", ref_text=ref_text, x_vector_only_mode=False)
for it in mine:
    t = time.time()
    wavs, sr = m.generate_voice_clone(text=it["text"], language="German", voice_clone_prompt=prompt, max_new_tokens=2048)
    sf.write(f"out/vo/{it['id']}.wav", wavs[0], sr)
    print(it["id"], round(time.time() - t), "s ->", round(len(wavs[0]) / sr, 1), "s", flush=True)
