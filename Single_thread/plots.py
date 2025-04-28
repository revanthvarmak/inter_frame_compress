import matplotlib.pyplot as plt
import numpy as np

videos = ["Video 1\n(136×240)", "Video 2\n(426×240)", "Video 3\n(320×240)"]
compression = [432.689, 1519.92, 1114.44]
decompression = [0.758215, 2.40844, 1.79349]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[3, 1])

bars1 = ax1.bar(videos, compression, color='blue', width=0.4)
ax1.set_ylabel('Compression Time (ms)')
ax1.set_title('Video Processing Performance')

bars2 = ax2.bar(videos, decompression, color='red', width=0.4)
ax2.set_ylabel('Decompression Time (ms)')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 20, f'{height:.1f}', 
             ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('compression_times.png')
plt.show()