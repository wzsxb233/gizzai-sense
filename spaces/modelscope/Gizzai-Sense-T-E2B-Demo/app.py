import json, os
from functools import lru_cache
import gradio as gr

MODEL_ID = os.environ.get("MODEL_ID", "GizzAI/Gizzai-Sense-T-E2B")

@lru_cache(maxsize=1)
def load_model():
    import torch
    from modelscope import snapshot_download
    path = snapshot_download(MODEL_ID)
    import sys; sys.path.insert(0, path)
    from sense_t import SenseT
    return SenseT(path, device="cuda:0" if torch.cuda.is_available() else "cpu")

def forecast(series_text, state, question):
    try:
        series = json.loads(series_text)
        s = load_model()
        out = {"forecast": s.forecast(series, step=2, horizon=7, state=state or "")}
        if question.strip(): out["decision"] = s.noul(state or "", question, series, horizon=7)
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

css = open("design/design_system.css", encoding="utf-8").read()
sample = json.dumps({"names":["门店A 日销量 / Store A daily sales"],"values":[[980,1012,1040,995,1030,1105,1220,970,1008,1035,1001,1042,1110,1235,965,1015,1030,999,1038,1102,1228]],"unit":"件 / units","step":"天 / day"}, ensure_ascii=False, indent=2)
with gr.Blocks(css=css, title="GizzAI Sense-T / 叽喳 Sense-T") as demo:
    gr.Markdown(f"""<div class='gz-panel'><span class='gz-badge'>LIVE DEMO / 在线试用</span><h1 class='gz-brand'>叽喳 Sense-T · GizzAI Sense-T</h1><p class='gz-note'>时序判断与 10/50/90% 预测 / Time-series decisions and 10/50/90% forecasts · <b>{MODEL_ID}</b></p><p class='gz-note'>首次加载会下载模型，可能需要几分钟。 / The first request downloads the checkpoint and may take several minutes.</p></div>""")
    series = gr.Code(label="时序 JSON / Series JSON", language="json", value=sample, lines=14)
    state = gr.Textbox(label="状态 / State", value="未来第 7 天有促销。 / Promotion on day 7.")
    question = gr.Textbox(label="判断问题（可留空） / Decision question (optional)", value="未来 7 天会不会有一天低于 950 件？ / Will any of the next 7 days be below 950 units?")
    run = gr.Button("运行 Sense-T / Run Sense-T", elem_classes=["gz-button"], variant="primary")
    output = gr.JSON(label="结果 / Result")
    run.click(forecast, [series, state, question], output)
    gr.Markdown("<div class='gz-footer'>GizzAI Sense License · 预测区间不是保证；重要事项请保留人工复核。 / Forecast bands are not guarantees; keep a person in the loop for material decisions.</div>")

demo.queue().launch()
