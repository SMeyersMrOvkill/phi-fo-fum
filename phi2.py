from together import Together
import os
import modal
import fastapi_poe as fp
from typing import AsyncIterable
import json

image = modal.Image.debian_slim().pip_install(['together', 'fastapi_poe>=0.0.28', 'modal'])
app = modal.App("dcw-phi-2-togcog", image=image)

def infer():
    pass

def togcog(messages):
    tg = Together(api_key=os.getenv("TOGETHER_KEY"))
    hist = [{'role': row.role, 'content': row.content} for row in messages]
    if len(hist) < 3:
        summary = "A new conversation between a human and an AI assistant."
        summary = tg.chat.completions.create(
        messages=[{'role': 'user', 'content': f"""Written below is a conversation as a JSON map. Condense it to bare essentials, cut out repeated details, generally compress. Each tripled newline is a row, each triple hyphen is a column. Format your output as Markdown.\n"""},
     {'role': 'user', 'content': json.dumps(hist)}],
        model="Qwen/Qwen1.5-1.8B-Chat")
        hist.insert(0, {'role': 'system', 'content': summary.choices[0].message.content})
    prompt = """
SYSTEM: Behave as a helpful, friendly, cooperative, morally distinguished assistant.
"""
    prompt += "\n\n\nUSER: " + messages[-1].content + "\n\n\nASSISTANT:"
    print(prompt)
    res = tg.completions.create(
        prompt=prompt,
        model="microsoft/phi-2",
        stop=["###", "\n\n\n", "ASSISTANT:","USER:","SYSTEM:"],
        max_tokens=2048-len(prompt)
    )
    combined = prompt + res.choices[0].text
        
    return res

class Phi2Bot(fp.PoeBot):
    async def get_response(
    self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        yield fp.PartialResponse(text="Calculating response...", is_replace_response=True)
        val = togcog(request.query)
        yield fp.PartialResponse(text=val.choices[0].text, is_replace_response=True)        
        return

@app.function(image=image,secrets=[modal.Secret.from_name("together"), modal.Secret.from_name("phis-darkest-dreams")])
@modal.asgi_app()
def fastapi_app():
    return fp.make_app(Phi2Bot(), access_key=os.getenv("PHI2_ACCESS_KEY"))