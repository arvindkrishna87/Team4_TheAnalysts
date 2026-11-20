from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("/home/wtq4672/Presentation3_base_code")
OUTPUT_DIR = BASE_DIR / "misclassification_analysis"

# Read the predictions from the 50 completed runs
df = pd.read_csv(OUTPUT_DIR / "all_930_predictions.csv")

# Create a case-by-repetition table
table = df.pivot(
    index="case_id",
    columns="repetition",
    values="misclassified"
)

# Keep only cases misclassified at least once
table = table.loc[table.sum(axis=1) > 0]

# Sort from most to least frequently misclassified
table["Wrong"] = table.sum(axis=1)
table = table.sort_values("Wrong", ascending=False)

# Format the table to match the professor's example
display_table = table.copy()

for rep in range(1, 11):
    display_table[rep] = display_table[rep].map({1: "X", 0: ""})

display_table["Wrong"] = display_table["Wrong"].astype(str) + "/10"
display_table.index.name = "Case"

# Save a CSV version
display_table.to_csv(OUTPUT_DIR / "misclassification_table.csv")

# Create a presentation-ready image
fig, ax = plt.subplots(
    figsize=(10, len(display_table) * 0.32 + 0.5)
)
ax.axis("off")

column_labels = ["Case"] + [str(i) for i in range(1, 11)] + ["Wrong"]

cell_text = [
    [str(case_id)] + row.tolist()
    for case_id, row in display_table.iterrows()
]

plot_table = ax.table(
    cellText=cell_text,
    colLabels=column_labels,
    cellLoc="center",
    loc="center"
)

plot_table.auto_set_font_size(False)
plot_table.set_fontsize(12)
plot_table.scale(1, 1.6)

# Style the table
for (row, col), cell in plot_table.get_celld().items():
    cell.set_edgecolor("#DDDDDD")
    cell.set_linewidth(0.4)

    if row == 0:
        cell.set_text_props(weight="bold", color="black")
        cell.set_facecolor("#F0F0F0")
    elif col == 11:
        cell.set_text_props(weight="bold", color="#5B2C90")
    elif cell.get_text().get_text() == "X":
        cell.set_text_props(weight="bold", color="#5B2C90")

ax.set_title(
    "Misclassified Cases Across 10 Repetitions",
    fontsize=17,
    weight="bold",
    pad=20
)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig(
    OUTPUT_DIR / "misclassification_table.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)

plt.close()

print(f"Table created for {len(display_table)} misclassified cases!")
print(f"Saved to: {OUTPUT_DIR / 'misclassification_table.png'}")
print(f"CSV saved to: {OUTPUT_DIR / 'misclassification_table.csv'}")