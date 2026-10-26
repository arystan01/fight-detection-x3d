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

## Limitations

- The model only sees the center square of the video. A fight at the edge of the frame can be missed.
- Crowds, hugging, sports and people moving fast close to each other can give false alarms.
- On your own cameras it can work worse than on the validation videos.

## License

Free for personal, hobby and research use. Commercial use needs a paid license, contact me on GitHub ([@arystan01](https://github.com/arystan01)). See [LICENSE](LICENSE).
