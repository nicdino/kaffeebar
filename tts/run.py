import asyncio, json, os, edge_tts
os.makedirs("out", exist_ok=True)
jobs = json.load(open("tts/jobs.json"))
async def one(j):
    c = edge_tts.Communicate(j["text"], j["voice"], rate=j.get("rate", "-5%"), pitch=j.get("pitch", "+0Hz"))
    await c.save(f"out/{j['name']}.mp3")
async def main():
    for j in jobs:
        for k in range(4):
            try:
                await one(j); break
            except Exception as e:
                print("retry", j["name"], e); await asyncio.sleep(3)
asyncio.run(main())
print(os.listdir("out"))
