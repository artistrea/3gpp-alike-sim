from simulator.parameters import PropagationModelEnum, ScenarioParams
import matplotlib.pyplot as plt
import numpy as np

# fig_std, ax_std = plt.subplots()
# plt.title(r"Desvio Padrão de $\sigma_\tau$")
# plt.xlabel("f [GHz]")
# plt.ylabel(r"Desvio Padrão [s]")
# plt.tight_layout()

# fig_mean, ax_mean = plt.subplots()
# plt.title(r"Média de $\sigma_\tau$")
# plt.xlabel("f [GHz]")
# plt.ylabel(r"Média [$\mu$s]")
# plt.tight_layout()

f_c = np.linspace(0.5, 100, 1000)

# for propag in PropagationModelEnum:
for propag in []:
    print("propag", propag)
    for los in [False, True]:
        print("los", los)
        mean, std = ScenarioParams._get_delay_spread_stats(
          f_c, 
          propag,
          los
        )
        linestyle = "--" if propag is PropagationModelEnum.UMa else "-"
        prop_name = "UMa" if propag is PropagationModelEnum.UMa else "UMi"
        clr = "blue" if los else "red"
        label = f"{prop_name} {'LoS' if los else 'NLoS'}"
        ax_mean.plot(
          f_c,
          10**(mean) * 1e6,
          label=label,
          color=clr,
          linestyle=linestyle
        )
        ax_std.plot(
          f_c,
          10**(std),
          label=label,
          color=clr,
          linestyle=linestyle
        )

# ax_mean.legend()
# ax_mean.set_xscale("log")
# ax_mean.xaxis.grid(True, which='major', linestyle='-', linewidth=1, alpha=0.5)
# ax_mean.xaxis.grid(True, which='minor', linestyle=':', linewidth=1, alpha=0.3)
# ax_mean.set_xlim((0.5, 100))

# ax_std.legend()
# ax_std.set_xscale("log")
# ax_std.xaxis.grid(True, which='major', linestyle='-', linewidth=1, alpha=0.5)
# ax_std.xaxis.grid(True, which='minor', linestyle=':', linewidth=1, alpha=0.3)
# # ax_std.set_xlim((0.5, 100))
# plt.show()

fig_std, ax_std = plt.subplots()
# plt.title(r"Desvio Padrão de $\sigma_\theta$")
plt.title(r"Desvio Padrão de $\sigma_\phi$")
plt.xlabel("f [GHz]")
plt.ylabel(r"Desvio Padrão [deg]")
plt.tight_layout()

fig_mean, ax_mean = plt.subplots()
# plt.title(r"Média de $\sigma_\theta$")
plt.title(r"Média de $\sigma_\phi$")
plt.xlabel("f [GHz]")
plt.ylabel(r"Média [deg]")
plt.tight_layout()

f_c = np.linspace(0.5, 100, 1000)

for propag in PropagationModelEnum:
# for propag in [PropagationModelEnum.UMi]:
    print("propag", propag)
    for los in [False, True]:
        print("los", los)
        # mean, std = ScenarioParams._get_aoa_azim_stats(
        mean, std = ScenarioParams._get_aoa_elev_stats(
          f_c, 
          propag,
          los
        )
        # print("mean", mean)
        linestyle = "--" if propag is PropagationModelEnum.UMa else "-"
        prop_name = "UMa" if propag is PropagationModelEnum.UMa else "UMi"
        clr = "blue" if los else "red"
        label = f"{prop_name} {'LoS' if los else 'NLoS'}"
        ax_mean.plot(
          f_c,
          10**(mean),
          label=label,
          color=clr,
          linestyle=linestyle
        )
        ax_std.plot(
          f_c,
          10**(std),
          label=label,
          color=clr,
          linestyle=linestyle
        )

ax_mean.legend()
ax_mean.set_xscale("log")
ax_mean.xaxis.grid(True, which='major', linestyle='-', linewidth=1, alpha=0.5)
ax_mean.xaxis.grid(True, which='minor', linestyle=':', linewidth=1, alpha=0.3)
ax_mean.set_xlim((0.5, 100))

ax_std.legend()
ax_std.set_xscale("log")
ax_std.xaxis.grid(True, which='major', linestyle='-', linewidth=1, alpha=0.5)
ax_std.xaxis.grid(True, which='minor', linestyle=':', linewidth=1, alpha=0.3)
ax_std.set_xlim((0.5, 100))
plt.show()


