"""Hierarchical clustering of the 12 non-neutral emotion_prob_* columns
(classify_usvsthem.py's output) by how they co-occur with each other --
usvsthem_score and Emotions_Neutral are excluded, per the question this
answers: which emotions tend to show up together, independent of the
primary score or the "nothing in particular" bucket.

Distance between two emotions is 1 - their Pearson correlation (0 for
perfectly co-occurring, 1 for unrelated, 2 for perfect mirror-opposites),
so anti-correlated emotions end up far apart in the clustering rather than
being treated as "similar" the way 1-|r| would. Average linkage is used
since it works directly with this distance matrix without assuming it's
Euclidean (unlike Ward's method, which technically shouldn't be applied to
an arbitrary distance matrix like this one).

Menu of what happens when you run this file, in order:
  1. Load --input (debate_usvsthem_scores.csv), take the 12 non-neutral
     emotion_prob_* columns.
  2. Build the 1-r distance matrix and run average-linkage hierarchical
     clustering on it.
  3. Sweep every k from 2 to --max-k and report each one's silhouette
     score -- a quantitative answer to "how many clusters" instead of
     just eyeballing the dendrogram, since --num-clusters alone doesn't
     tell you whether that count is actually a good fit. Also saved as a
     plot (silhouette vs k).
  4. Cut the resulting tree into --num-clusters flat groups (printing a
     note if a different k scored better in step 3) and print which
     emotions fall into each.
  5. Save a dendrogram (shows the full merge hierarchy, not just the flat
     cut -- where an item joins matters as much as which side it ends up
     on) and a correlation heatmap reordered to match the dendrogram's
     leaf order, so visually adjacent emotions are the ones that cluster.

Usage:
    python usvsthem_emotion_clustering.py --input debate_usvsthem_scores.csv --num-clusters 3
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

EXCLUDED_EMOTION_COLUMNS = ["emotion_prob_Emotions_Neutral"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="debate_usvsthem_scores.csv")
    parser.add_argument("--num-clusters", type=int, default=3, help="How many flat clusters to cut the dendrogram into")
    parser.add_argument("--max-k", type=int, default=8, help="Sweep silhouette score over k=2..max-k to help pick --num-clusters")
    parser.add_argument("--dendrogram-plot", default="emotion_dendrogram.png")
    parser.add_argument("--heatmap-plot", default="emotion_clustered_heatmap.png")
    parser.add_argument("--silhouette-plot", default="emotion_silhouette_scores.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    emotion_columns = [
        c for c in df.columns
        if c.startswith("emotion_prob_") and c not in EXCLUDED_EMOTION_COLUMNS
    ]
    missing_check = [c for c in df.columns if c.startswith("emotion_prob_")]
    if not missing_check:
        raise SystemExit(f"--input has no emotion_prob_* columns -- was it produced by classify_usvsthem.py?")

    short_names = [c.removeprefix("emotion_prob_") for c in emotion_columns]
    print(f"Loaded {len(df)} row(s) from {args.input}. Clustering {len(emotion_columns)} emotion(s): {short_names}")

    corr = df[emotion_columns].corr(method="pearson")
    corr.index = short_names
    corr.columns = short_names

    distance = 1 - corr
    # Clean up any floating-point asymmetry/negative-near-zero before
    # squareform (which requires an exactly symmetric, zero-diagonal matrix)
    distance = (distance + distance.T) / 2
    for name in short_names:
        distance.loc[name, name] = 0.0
    condensed = squareform(distance.values, checks=False)

    link = linkage(condensed, method="average")

    # How many clusters actually fit the data well? Silhouette score answers
    # this quantitatively for each candidate k: +1 means every item is far
    # closer to its own cluster than any other, 0 means clusters overlap,
    # negative means items are probably in the wrong cluster. This is a
    # direct answer to "how do I know the right number of clusters" rather
    # than just eyeballing the dendrogram -- though the dendrogram is still
    # worth checking too, since silhouette can't tell you WHY a k is good.
    max_k = min(args.max_k, len(short_names) - 1)
    silhouette_results = []
    for k in range(2, max_k + 1):
        labels_k = fcluster(link, t=k, criterion="maxclust")
        if len(set(labels_k)) < 2:
            continue  # degenerate: this k collapsed to fewer actual groups
        score = silhouette_score(distance.values, labels_k, metric="precomputed")
        silhouette_results.append({"k": k, "silhouette": score})

    if silhouette_results:
        silhouette_df = pd.DataFrame(silhouette_results)
        best_row = silhouette_df.loc[silhouette_df["silhouette"].idxmax()]
        print(f"\nSilhouette score by number of clusters (k=2..{max_k}; higher is better, max +1):")
        print(silhouette_df.to_string(index=False, formatters={"silhouette": "{:.4f}".format}))
        print(f"\nBest-scoring k: {int(best_row['k'])} (silhouette={best_row['silhouette']:.4f})")
        if int(best_row["k"]) != args.num_clusters:
            print(
                f"NOTE: this differs from --num-clusters={args.num_clusters} -- the cut below still uses "
                f"--num-clusters as requested, but k={int(best_row['k'])} scored better. Consider rerunning "
                f"with --num-clusters {int(best_row['k'])} if you want the flat cut to match."
            )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(silhouette_df["k"], silhouette_df["silhouette"], marker="o")
        ax.axvline(args.num_clusters, color="gray", linestyle="--", linewidth=1, label=f"--num-clusters ({args.num_clusters})")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Cluster count vs. silhouette score (higher = better-separated)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.silhouette_plot, dpi=150)
        print(f"Saved silhouette-vs-k plot to {args.silhouette_plot}")
    else:
        print("\nCouldn't compute silhouette scores (not enough emotions to try more than one k).")

    cluster_ids = fcluster(link, t=args.num_clusters, criterion="maxclust")
    print(f"\n{args.num_clusters} cluster(s):")
    clusters = {}
    for name, cid in zip(short_names, cluster_ids):
        clusters.setdefault(cid, []).append(name)
    for cid in sorted(clusters):
        print(f"  Cluster {cid}: {', '.join(clusters[cid])}")

    fig, ax = plt.subplots(figsize=(9, 6))
    dendro = dendrogram(link, labels=short_names, ax=ax, color_threshold=link[-(args.num_clusters - 1), 2] if args.num_clusters > 1 else None)
    ax.set_ylabel("Distance (1 - Pearson r)")
    ax.set_title(f"Emotion clustering dendrogram (n={len(df)} item(s))")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.dendrogram_plot, dpi=150)
    print(f"\nSaved dendrogram to {args.dendrogram_plot}")

    # Reorder the correlation heatmap to match the dendrogram's leaf order,
    # so visually adjacent rows/columns are the ones the clustering grouped.
    leaf_order = dendro["ivl"]
    corr_reordered = corr.loc[leaf_order, leaf_order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_reordered.values, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(leaf_order)))
    ax.set_xticklabels(leaf_order, rotation=45, ha="right")
    ax.set_yticks(range(len(leaf_order)))
    ax.set_yticklabels(leaf_order)
    for i in range(len(leaf_order)):
        for j in range(len(leaf_order)):
            value = corr_reordered.values[i, j]
            text_color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation heatmap, reordered to match clustering")
    fig.tight_layout()
    fig.savefig(args.heatmap_plot, dpi=150)
    print(f"Saved clustering-ordered heatmap to {args.heatmap_plot}")


if __name__ == "__main__":
    main()
