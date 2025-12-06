import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Load image ---
img = np.array(Image.open("otter.jpg").convert("RGB"), dtype=float)
img /= 255.0  # normalize to [0,1]

h, w, c = img.shape

CH = [0] * 3
U = [0] * 3
S = [0] * 3
Vt = [0] * 3

# precompute
for ch in range(3):
    CH[ch] = img[:,:,ch]

    # SVD-based PCA
    U[ch], S[ch], Vt[ch] = np.linalg.svd(CH[ch], full_matrices=False)


mse_values = []
cr_values = []

plt.ion()

target_width_px = w - 200
dpi = 100                   # default DPI
fig_width_in = target_width_px / dpi
fig_height_in = 5           # choose any height in inches

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

line, = ax.plot([], [], label='MSE')
ax.set_xlim(1, 200)
ax.set_ylim(0, 0.05)
ax.set_xlabel("k")
ax.set_ylabel("MSE")
ax.set_title("Real-time PCA Reconstruction Error")
ax.grid(True)

ax2 = ax.twinx()
line_cr, = ax2.plot([], [], color="orange", label="Compression ratio")
ax2.set_ylabel("Compression ratio")
ax2.set_ylim(0, 100)

fig2, axes = plt.subplots(2, 1, figsize=(w/100, 2*h/100))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

recon_im = axes[1].imshow(np.zeros_like(img))
axes[1].set_title("Reconstructed")
axes[1].axis("off")
plt.tight_layout()
plt.show()

k_values = range(1, 201)   # test k = 1..100

plt.waitforbuttonpress()

for k in k_values:
    recon = np.zeros_like(img)

    for ch in range(3):
        # keep only top-k singular values
        Uk = U[ch][:, :k]
        Sk = np.diag(S[ch][:k])
        Vk = Vt[ch][:k, :]

        # reconstruct channel
        Xk = Uk @ Sk @ Vk
        recon[:,:,ch] = Xk

    # clip + save
    recon = np.clip(recon, 0, 1)

    # Mean squared error over all pixels and channels
    mse = np.mean((img - recon) ** 2)
    mse_values.append(mse)

    cr = (h * w) / (k * (h + w + 1))
    cr_values.append(cr)

    print(f"k={k}/{h} mse={mse:.6f} compression={cr:.6f}")   # 6 decimal places

    # Update real-time plot
    line.set_xdata(list(k_values)[:len(mse_values)])
    line.set_ydata(mse_values)

    line_cr.set_xdata(list(k_values)[:len(cr_values)])
    line_cr.set_ydata(cr_values)

    # ax.relim()       # recompute data limits (optional)
    # ax.autoscale()   # autoscale the view (optional)

    plt.draw()
    plt.pause(0.001)

    # --- Update reconstruction plot ---
    recon_im.set_data(recon)
    fig2.canvas.draw_idle()  # update figure
    plt.pause(0.001)

# --- After all plotting is done ---
plt.ioff()      # turn off interactive mode
plt.show()      # keep the window open until manually closed

