import sys

from model import check_video, load_model

model, device = load_model()

for path in sys.argv[1:]:
    times, scores, alert_at = check_video(model, device, path)
    if alert_at is not None:
        print(f"{path}: FIGHT at {alert_at:.1f}s (max {max(scores):.2f})")
    else:
        print(f"{path}: no fight (max {max(scores):.2f})")
