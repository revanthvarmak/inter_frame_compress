import matplotlib.pyplot as plt

# ── raw timing data ────────────────────────────────────────────────
videos = ["136×240", "320×240", "426×240"]

cpu_avg_comp    = [1899.8, 4728.96, 6335.25]      # ms / frame
cpu_avg_decomp  = [2.76038, 6.4604, 8.54502]
cpu_frames      = [301, 602, 601]

cuda_avg_comp   = [2.63524, 3.78238, 4.48372]
cuda_avg_decomp = [0.325957, 0.440143, 0.518017]
cuda_frames     = [301, 602, 601]

# ── totals & speed-ups ─────────────────────────────────────────────
cpu_total_comp    = [a*f for a, f in zip(cpu_avg_comp,    cpu_frames)]
cpu_total_decomp  = [a*f for a, f in zip(cpu_avg_decomp,  cpu_frames)]
cuda_total_comp   = [a*f for a, f in zip(cuda_avg_comp,   cuda_frames)]
cuda_total_decomp = [a*f for a, f in zip(cuda_avg_decomp, cuda_frames)]

speedup_comp   = [c/g for c, g in zip(cpu_total_comp,   cuda_total_comp)]
speedup_decomp = [c/g for c, g in zip(cpu_total_decomp, cuda_total_decomp)]

# ── plotting helper ────────────────────────────────────────────────
def plot_totals(title, cpu_vals, gpu_vals, speedups, ylabel, outfile,
                log_y=False):
    x, w = range(len(videos)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i-w/2 for i in x], cpu_vals, w, label="CPU (single-threaded)")
    ax.bar([i+w/2 for i in x], gpu_vals, w, label="CUDA (multi-threaded)")
    ax.set_xticks(x, videos)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Video resolution")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend()

    # annotate speed-up above CPU bar
    for i, (cpu_v, s) in enumerate(zip(cpu_vals, speedups)):
        ax.text(i-w/2, cpu_v, f"{s:.1f}×", ha="center",
                va="bottom", fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(f"{outfile}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outfile}.pdf",            bbox_inches="tight")

# ── build & save figures ───────────────────────────────────────────
plot_totals("Total Compression Time per Video",
            cpu_total_comp, cuda_total_comp, speedup_comp,
            "Total compression time (ms)",
            "compression_totals", log_y=True)

plot_totals("Total Decompression Time per Video",
            cpu_total_decomp, cuda_total_decomp, speedup_decomp,
            "Total decompression time (ms)",
            "decompression_totals", log_y=False)

plt.show()
