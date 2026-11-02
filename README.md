# Fight detection

A small model that looks at a video and tells if there is a fight in it.

**[▶ Try it online](https://huggingface.co/spaces/Arystank01/fight-detection)** — upload a video and check it right in your browser.

## Run it

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps pytorchvideo==0.1.5

python app.py
```

Open http://127.0.0.1:7860, upload a video, press **Check**.

Or from the terminal:

```bash
python predict.py video.mp4
```

Works on CPU, uses GPU if you have one.

## How it works

The model looks at 2 seconds of video at a time (16 frames) and gives a fight score from 0 to 1.
It does this every 0.5 seconds. If 3 of the last 5 scores are above 0.6, it says there is a fight.

## Training

- Model: X3D-M (about 3M parameters), pretrained on Kinetics-400
- Data: 3000+ short videos from CCTV and phones, about half with fights, half without
- Each training sample is a 2 second piece of a video, 16 frames, 224x224
- Best result after 3 epochs

To train it on your own videos, put them in folders like `data/train/Fight/*.avi`, `data/train/NonFight/*.avi`, `data/val/Fight/*.avi`, `data/val/NonFight/*.avi` and run:

```bash
python train.py data
```

## Results

On 400 validation videos:

| | Precision | Recall | F1 |
|---|---|---|---|
| Fight | 88.8% | 87.5% | 88.2% |
| No fight | 87.7% | 89.0% | 88.3% |

**Accuracy: 88.25%**

| | predicted no fight | predicted fight |
|---|---|---|
| no fight | 178 | 22 |
| fight | 25 | 175 |

## Agent

The model alone gives a lot of false alarms on crowds, hugs, sports etc. So there is a second step: when X3D thinks there is a fight, a small vision language model looks at 8 frames and says if it's really a fight and what is happening. If yes, the clip goes to Telegram.

```
camera -> X3D (every 0.5s) -> fight? -> Qwen3.5-2B looks at frames -> fight? -> Telegram clip + description
```

X3D here looks at 3 squares of the frame (left, center, right) instead of only the center and uses a lower threshold (0.3), so it misses less. The VLM removes the extra false alarms.

Run the VLM (needs vllm, GPU with 8GB+):

```bash
pip install vllm
bash vlm.sh                     # base Qwen/Qwen3.5-2B
bash vlm.sh path/to/finetuned   # or your fine-tuned one
```

Then run the agent on a video or a camera:

```bash
export TG_TOKEN=...   # optional, telegram bot
export TG_CHAT=...
python agent.py video.mp4
python agent.py rtsp://user:pass@camera-ip/stream rtsp://...
```

Every decision is saved to `alerts.jsonl`, clips to `alerts/`.

### Fine-tuning the VLM

The base 2B model rejects too many real fights. Fine-tuning it on the same fight / no fight videos fixes that. Takes about 40 min on a RTX 2080 Ti.

```bash
pip install peft transformers
python train_vlm.py data/train my-vlm
bash vlm.sh my-vlm
python eval_agent.py data/val
```

### Agent results

On the same 400 validation videos:

| | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|
| X3D only (center, threshold 0.6) | 91.0% | 85.5% | 88.1% | 17 |
| X3D (3 squares, 0.3) | 76.9% | 98.0% | 86.2% | 59 |
| X3D + base Qwen3.5-2B | 85.2% | 97.5% | 90.9% | 34 |
| X3D + fine-tuned Qwen3.5-2B | **92.6%** | **94.5%** | **93.6%** | 15 |

The VLM answers in ~0.6s on a RTX 2080 Ti.

## Limitations

- The model only sees the center square of the video. A fight at the edge of the frame can be missed.
- Crowds, hugging, sports and people moving fast close to each other can give false alarms.
- On your own cameras it can work worse than on the validation videos.

## License

Free for personal, hobby and research use. Commercial use needs a paid license, contact me on GitHub ([@arystan01](https://github.com/arystan01)). See [LICENSE](LICENSE).
