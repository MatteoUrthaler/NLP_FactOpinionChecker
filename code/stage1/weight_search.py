import subprocess
import re
from itertools import product

pos_weight_grid = [0.8, 1.0, 1.2]
ner_weight_grid = [1.0, 1.4, 1.8, 2.0]
emotion_weight_grid = [0.6, 0.8, 1.0]
# should be expanded to include the other two as well but the computation time is insane

results = []

for pos_w, ner_w, emo_w in product(pos_weight_grid, ner_weight_grid, emotion_weight_grid):
    result = subprocess.run(
        [
            "python3",
            "code/stage1/combined_features.py",
            str(pos_w),
            str(ner_w),
            str(emo_w),
        ],
        capture_output=True,
        text=True,
        check=True
    )

    stdout = result.stdout

    mean_match = re.search(r"Mean accuracy:\s*([0-9.]+)%", stdout)
    std_match = re.search(r"Standard deviation:\s*([0-9.]+)", stdout)

    if mean_match and std_match:
        mean_accuracy = float(mean_match.group(1))
        std_accuracy = float(std_match.group(1))

        results.append({
            "pos_weight": pos_w,
            "ner_weight": ner_w,
            "emotion_weight": emo_w,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
        })

        print(
            f"POS {pos_w}, NER {ner_w}, EMOTION {emo_w} "
            f"  Mean {mean_accuracy:.2f}% Std {std_accuracy:.2f}"
        )
    else:
        print("error")

results.sort(key=lambda x: (x["mean_accuracy"], -x["std_accuracy"]), reverse=True)

best = results[0]

print("\nbest!:")
print(best)