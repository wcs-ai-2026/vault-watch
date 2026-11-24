import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `vaultwatch` imports work when run as a script
from vaultwatch.models.tune_xgb import TARGET_RECALL, load_splits  # noqa: E402
from vaultwatch.models.xgb import FEATURE_COLUMNS, XGBModel  # noqa: E402

# VW-31: feature importance report and precision/recall vs threshold curve for XGBoost.
#
# Trains XGBModel (default settings, chosen in VW-30) on the training split and scores the
# validation split. Writes two charts and a markdown summary to docs/. The test split is
# never loaded.
#
#   python src/vaultwatch/models/xgb_report.py --data full   # what docs/ was made with
#   python src/vaultwatch/models/xgb_report.py --data seed   # quick version, noisy (11 fraud rows)

FIGURES_DIR = Path("docs/figures")
REPORT_PATH = Path("docs/xgb_report.md")
REPORT_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

# chart colours (light surface, validated palette slots 1-2)
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

FEATURE_LABELS = {
    "amount": "Amount",
    "oldbalanceOrg": "Sender balance before",
    "newbalanceOrig": "Sender balance after",
    "oldbalanceDest": "Receiver balance before",
    "newbalanceDest": "Receiver balance after",
    "type_CASH_IN": "Type: CASH_IN",
    "type_CASH_OUT": "Type: CASH_OUT",
    "type_DEBIT": "Type: DEBIT",
    "type_PAYMENT": "Type: PAYMENT",
    "type_TRANSFER": "Type: TRANSFER",
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_feature_importance(importance, path):
    share = importance / importance.sum()  # gain as a share of the total, easier to read than raw gain
    share = share.iloc[::-1]  # biggest at the top
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style(ax)
    ax.barh([FEATURE_LABELS.get(n, n) for n in share.index], share.values, color=BLUE, height=0.6)
    for y, value in enumerate(share.values):
        ax.text(value + 0.005, y, f"{value:.0%}", va="center", fontsize=9, color=TEXT)
    ax.set_xlabel("Share of total gain (how much each feature improved the trees' splits)", color=MUTED, fontsize=9)
    ax.set_title("XGBoost: which features matter most", loc="left", color=TEXT, fontsize=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, share.max() * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(precision, recall, thresholds, chosen_threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style(ax)
    # precision_recall_curve returns one more precision/recall value than thresholds
    ax.plot(thresholds, precision[:-1], color=BLUE, linewidth=2, label="Precision (alerts that are real fraud)")
    ax.plot(thresholds, recall[:-1], color=ORANGE, linewidth=2, label="Recall (fraud that gets caught)")
    ax.axhline(TARGET_RECALL, color=MUTED, linewidth=1, linestyle=":")
    ax.text(0.99, TARGET_RECALL + 0.015, f"{TARGET_RECALL:.0%} recall goal", color=MUTED, fontsize=8, ha="right")
    ax.axvline(chosen_threshold, color=MUTED, linewidth=1, linestyle="--")
    ax.text(chosen_threshold + 0.015, 0.03, f"highest threshold\nkeeping {TARGET_RECALL:.0%} recall: {chosen_threshold:.2f}",
            color=MUTED, fontsize=8, ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Alert threshold (flag a transaction when its score is at least this)", color=MUTED, fontsize=9)
    ax.set_title("XGBoost on validation: precision and recall vs threshold", loc="left", color=TEXT, fontsize=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", frameon=False, fontsize=9, labelcolor=TEXT, bbox_to_anchor=(0.99, 0.4))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["seed", "full"], default="seed")
    args = parser.parse_args()

    train, val = load_splits(args.data)
    model = XGBModel().fit(train[FEATURE_COLUMNS], train["isFraud"])
    scores = model.predict_score(val[FEATURE_COLUMNS])
    labels = val["isFraud"].to_numpy()

    importance = model.feature_importance()
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # highest threshold that still catches TARGET_RECALL of the fraud (fewest false alarms at that recall)
    keeps_recall = recall[:-1] >= TARGET_RECALL
    chosen_threshold = float(thresholds[keeps_recall].max())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(importance, FIGURES_DIR / "xgb_feature_importance.png")
    plot_threshold_curve(precision, recall, thresholds, chosen_threshold, FIGURES_DIR / "xgb_threshold_curve.png")

    # summary table at a few thresholds
    rows = []
    for threshold in REPORT_THRESHOLDS + [chosen_threshold]:
        flagged = scores >= threshold
        caught = int((flagged & (labels == 1)).sum())
        rows.append({
            "threshold": round(threshold, 3),
            "alerts": int(flagged.sum()),
            "fraud caught": f"{caught} / {int(labels.sum())}",
            "precision": round(caught / flagged.sum(), 3) if flagged.any() else 0.0,
            "recall": round(caught / labels.sum(), 3),
        })
    table = pd.DataFrame(rows).drop_duplicates("threshold").sort_values("threshold")
    share = (importance / importance.sum()).round(3)

    lines = [
        "# XGBoost feature importance and threshold report (VW-31)",
        "",
        f"Generated by `python src/vaultwatch/models/xgb_report.py --data {args.data}`.",
        f"Trained on the training split (steps 1-323, {len(train):,} rows, {int(train['isFraud'].sum()):,} fraud) "
        f"with the VW-30 settings, scored on the validation split (steps 324-378, {len(val):,} rows, "
        f"{int(labels.sum()):,} fraud). The test split was not used.",
        "",
        f"**Validation PR-AUC: {average_precision_score(labels, scores):.3f}**",
        "",
        "## Which features matter most",
        "",
        "![Feature importance](figures/xgb_feature_importance.png)",
        "",
        "Share of total gain (how much each feature improved the trees' splits):",
        "",
        "| Feature | Share of gain |",
        "|---|---:|",
        *[f"| {FEATURE_LABELS.get(name, name)} (`{name}`) | {value:.1%} |" for name, value in share.items()],
        "",
        "## Precision and recall vs threshold",
        "",
        "![Threshold curve](figures/xgb_threshold_curve.png)",
        "",
        "| Threshold | Alerts | Fraud caught | Precision | Recall |",
        "|---:|---:|---:|---:|---:|",
        *[f"| {r['threshold']} | {r['alerts']:,} | {r['fraud caught']} | {r['precision']:.3f} | {r['recall']:.3f} |"
          for r in table.to_dict("records")],
        "",
        f"The highest threshold that still catches {TARGET_RECALL:.0%} of validation fraud is **{chosen_threshold:.3f}**. "
        "The final alert threshold belongs to the ensemble (VW-48), so this is input for that choice, not a decision.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {FIGURES_DIR}/xgb_feature_importance.png, {FIGURES_DIR}/xgb_threshold_curve.png and {REPORT_PATH}")


if __name__ == "__main__":
    main()
