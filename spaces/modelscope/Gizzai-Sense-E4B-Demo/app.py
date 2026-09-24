import json, os
from functools import lru_cache
import gradio as gr

MODEL_ID = os.environ.get("MODEL_ID", "GizzAI/Gizzai-Sense-E4B")
PLATFORM = os.environ.get("MODEL_PLATFORM", "Hugging Face")

@lru_cache(maxsize=1)
def load_model():
    import torch
    from modelscope import snapshot_download
    path = snapshot_download(MODEL_ID)
    import sys; sys.path.insert(0, path)
    from sense import Sense
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return Sense(path, device=device, dtype=dtype)

def decide(state, question, options):
    if not question.strip(): return {"error": "请输入问题 / Enter a question"}
    try:
        s = load_model()
        if options.strip():
            parsed = json.loads(options)
            out = s.choice(state or "", question, parsed, open_set=False)
        else:
            out = {"type": "noul", "noul": s.noul(state or "", question)}
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

css = open("design/design_system.css", encoding="utf-8").read()
with gr.Blocks(css=css, title="GizzAI Sense / 叽喳 Sense") as demo:
    gr.Markdown(f"""<div class='gz-panel'><span class='gz-badge'>LIVE DEMO / 在线试用</span><h1 class='gz-brand'>叽喳 Sense · GizzAI Sense</h1><p class='gz-note'>校准概率判断 / Calibrated typed decisions · <b>{MODEL_ID}</b></p><p class='gz-note'>首次加载会下载模型，可能需要几分钟。 / The first request downloads the checkpoint and may take several minutes.</p></div>""")
    with gr.Row():
        with gr.Column():
            state = gr.Textbox(label="状态 / State", lines=4, value="连续两次部署失败，客户看到 500 错误。 / Two deployments failed and customers saw 500 errors.")
            question = gr.Textbox(label="问题 / Question", value="需要立即处理吗？ / Should this be handled immediately?")
            options = gr.Textbox(label="可选项 JSON（留空为是/否） / Options JSON (blank = yes/no)", lines=3, value='{"infra":"Infrastructure", "billing":"Billing"}')
            run = gr.Button("判断 / Decide", elem_classes=["gz-button"], variant="primary")
        output = gr.JSON(label="校准结果 / Calibrated result")
    run.click(decide, [state, question, options], output)
    gr.Markdown("<div class='gz-footer'>GizzAI Sense License · 概率用于辅助决策；重要事项请保留人工复核。 / Probabilities support decisions; keep a person in the loop for material decisions.</div>")

demo.queue().launch()
