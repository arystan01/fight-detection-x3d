import gradio as gr
import pandas as pd

from model import THRESHOLD, check_video, load_model

model, device = load_model()


def run(video, threshold):
    if video is None:
        return "Upload a video", None
    try:
        times, scores, alert_at = check_video(model, device, video, threshold)
    except ValueError as e:
        return f"Error: {e}", None

    if alert_at is not None:
        text = f"## 🔴 Fight detected at {alert_at:.1f}s"
    else:
        text = "## 🟢 No fight"
    text += f"\n\nMax score: {max(scores):.2f}"
    return text, pd.DataFrame({"time": times, "fight": scores})


with gr.Blocks(title="Fight detection") as demo:
    gr.Markdown("# Fight detection")
    with gr.Row():
        with gr.Column():
            video = gr.Video(sources=["upload"])
            threshold = gr.Slider(0.3, 0.95, value=THRESHOLD, step=0.05, label="Threshold")
            button = gr.Button("Check", variant="primary")
        with gr.Column():
            result = gr.Markdown()
            plot = gr.LinePlot(x="time", y="fight", y_lim=[0, 1], label="Fight score")
    button.click(run, [video, threshold], [result, plot])

demo.launch()
