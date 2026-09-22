"""Download Gizzai Sense from Hugging Face and ask it three kinds of question.

    python examples/quickstart.py [--model GizzAI/Gizzai-Sense-E2B] [--device cuda]
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # sense.py lives at the repository root
from sense import Sense  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="GizzAI/Gizzai-Sense-E2B")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

path = snapshot_download(args.model)
s = Sense(path, device=args.device)

print("P(yes):", round(s.noul("部署连续失败两次，客户已经看到 500 错误。", "需要立即处理吗？"), 3))

team = s.choice("Checkout returns 502 for every EU customer since 09:00.", "Which team owns this?",
                {"infra": "Infrastructure", "billing": "Billing"}, open_set=True)
print("choice:", team["choice"], {k: round(v, 3) for k, v in team["probabilities"].items()})

mood = s.score("Arrived two weeks late and the box was crushed.", "How satisfied is the customer?",
               ["angry", "unhappy", "neutral", "happy"])
print("score:", round(mood["score"], 2), {k: round(v, 3) for k, v in mood["probabilities"].items()})
