import xml.etree.ElementTree as ET
import os
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import time

login(token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))


device = "cuda"

deltas = torch.load("photorealistic_deltas.pt")
deltas = {k: v.to(device).half() for k, v in deltas.items()}

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16
).to(device)

def genImage(prompt, height, width):
    with torch.no_grad():
        pe, ne, pp, np = pipe.encode_prompt(prompt, None, None, device=device, do_classifier_free_guidance=True)
        pe += deltas["pePCA"]
        ne += deltas["nePCA"]
        pp += deltas["ppPCA"]
        np += deltas["npPCA"]
        img = pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            pooled_prompt_embeds=pp,
            negative_pooled_prompt_embeds=np,
            height=height,
            width=width,
            num_inference_steps=30,
            guidance_scale=7.0
        ).images[0]
        return img

topics = [
    {
        "num": t.find("num").text,
        "title": t.find("title").text,
        "prompt": t.find("prompt").text
    }
    for t in ET.parse("round-1-topics.xml").findall("topic")
]

submission_dir = "my_submission"
image_dir = os.path.join(submission_dir, "images_generated")
prompt_dir = os.path.join(submission_dir, "images_prompts")
for d in (image_dir, prompt_dir):
    os.makedirs(d, exist_ok=True)

root = ET.Element("GeneratorResults", {"teamName": "GMU"})

run_result_el = ET.SubElement(
    root,
    "GeneratorRunResult",
    {
        "trainingData": "base/enriched prompt embedding deltas",
        "priority": "1",
        "trained": "T",
        "desc": "Encoder control vector run",
        "link": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large https://github.com/SurenSK/NIST",
    },
)
resolutions = [(1280, 560), (1280, 720), (1280, 960), (640, 480), (960, 720)]

for i, topic in enumerate(topics):
    print(f"Starting {i}")
    topic_num_full = topic["num"]
    prompt = topic["prompt"]
    topic_id = topic_num_full.split("_")[1]
    topic_el = ET.SubElement(run_result_el, "GeneratorTopicResult", {"topic": topic_num_full, "usedImagePrompts": "F"})
    topic_start = time.time()
    for idx, (w, h) in enumerate(resolutions, 1):
        filename = f"topic.{topic_id}.image.{idx}.webp"
        filepath = os.path.join(image_dir, filename)
        image = genImage(prompt, h, w)
        image.save(filepath, "webp")
        ET.SubElement(topic_el, "Image", {"filename": filename, "prompt": prompt, "NIST-prompt": "T"})
        del image
        torch.cuda.empty_cache()
    elapsedTime = time.time()-topic_start
    # The DTD requires the elapsedTime attribute to be set
    topic_el.set("elapsedTime", str(elapsedTime))
    print(f"Finished {i}")
tree = ET.ElementTree(root)
ET.indent(tree, space="    ")
xml_path = os.path.join(submission_dir, "submission.xml")
with open(xml_path, "wb") as f:
    f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write(b'<!DOCTYPE GeneratorResults SYSTEM "t2i_GeneratorResult.dtd">\n')
    tree.write(f, encoding="UTF-8")

print(f"\nSuccess! Submission created in the '{submission_dir}/' directory.")
