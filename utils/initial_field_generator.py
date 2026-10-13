# %% Cell
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %% Cell

dx = 0.1
N = 1024
L = N * dx
#
np.random.seed(42)  # heat the randomizer
lambda_min, lambda_max = L / 64.0, L / 8.0  # lambda range limits
num_waves = 32
lambdas = np.random.uniform(
    lambda_min, lambda_max, num_waves
)  # set of random wavelengths
k_values = 2.0 * np.pi / lambdas  # wavenumbers
phases = np.random.uniform(0, 2 * np.pi, num_waves)  # set of random phases
thetas = np.random.uniform(0, 2 * np.pi, num_waves)  # set of random angles
x = np.linspace(0, L, N)
X, Y = np.meshgrid(x, x)


def generate_2d_field(
    N=N, num_waves=num_waves, k_values=k_values, phases=phases, thetas=thetas
):
    u = np.zeros((N, N))
    for i in range(num_waves):
        arg = k_values[i] * (X * np.cos(thetas[i]) + Y * np.sin(thetas[i])) + phases[i]
        u += np.sin(arg)
    u = u / np.sqrt(num_waves)
    return u


u = generate_2d_field()

# %% Cell
# plot field
with plt.style.context("dark_background"):
    fig, ax = plt.subplots()
    im = ax.imshow(u, cmap="twilight_shifted", origin="lower", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, label="Value u")
    ax.set_xlabel("X, pixels")
    ax.set_ylabel("Y, pixels")
    plt.tight_layout()
    plt.show()

# plt.savefig('figure.svg', bbox_inches='tight')
# %% Cell
# plot section
plt.plot(u[:, (N - 1) // 2])
plt.show()
# %%
# save to hdf5 file
# with h5py.File("initial-file.h5", "w") as hf:
#     hf.create_dataset("u", data=u)
