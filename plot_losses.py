import json
import pathlib
import matplotlib.pyplot as plt

losses_dir = pathlib.Path(__file__).parent / 'losses'
files = sorted(losses_dir.glob('*.json'))

if not files:
    print("No loss files found. Run one or more models first.")
    exit(1)

fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Loss Curves by Model Architecture', fontsize=14)

for f in files:
    data = json.loads(f.read_text())
    steps = [d['step'] for d in data]
    train = [d['train'] for d in data]
    val = [d['val'] for d in data]
    ax_train.plot(steps, train, marker='o', label=f.stem)
    ax_val.plot(steps, val, marker='o', label=f.stem)

for ax, title in [(ax_train, 'Train Loss'), (ax_val, 'Val Loss')]:
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = pathlib.Path(__file__).parent / 'loss_curves.png'
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
