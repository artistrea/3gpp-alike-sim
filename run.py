from simulator import run_simulation
from simulator.parameters import ScenarioParams, PropagationModelEnum

import matplotlib.pyplot as plt
import numpy as np

def rect(dt, N, end):
    t = np.linspace(0, end * dt, N)
    tx = np.zeros_like(t)
    tx[0:N//end] = 1

    return (t, tx)


scenario = ScenarioParams(
    seed = 126,

    los = False,
    propagation_model = PropagationModelEnum.UMi,

    f_c = 3,
    ut_velocity = (5.0, 0.0, 0.0),
    ut_position = (0.0, 0.0, 0.0),
    bs_position = (10.0, 0.0, 10.0),
)

t_e3, tx_e3 = rect(1e-3, int(1e5), 5)
t_e5, tx_e5 = rect(1e-5, int(1e5), 5)
t_e7, tx_e7 = rect(1e-7, int(1e5), 15)
results_1_1 = run_simulation(
    scenario,
    tx_e3,
    t_e3,
)
results_2_1 = run_simulation(
    scenario,
    tx_e5,
    t_e5,
)
results_3_1 = run_simulation(
    scenario,
    tx_e7,
    t_e7,
)

scenario.ut_velocity.x = 50.0
results_1_2 = run_simulation(
    scenario,
    tx_e3,
    t_e3,
)
results_2_2 = run_simulation(
    scenario,
    tx_e5,
    t_e5,
)
results_3_2 = run_simulation(
    scenario,
    tx_e7,
    t_e7,
)
# print(np.max(results.clusters_delay))

print("Plotting power components in time delay")
fig, ax = plt.subplots()
ax.stem(results_1_1.clusters_delay * 1e6, results_1_1.clusters_power, markerfmt="^k", linefmt="k")
ax.stem([results_1_1.clusters_delay[0] * 1e6], [results_1_1.clusters_power[0]], markerfmt="^b", linefmt="blue")
ax.set_yscale("log")
ax.set_title("Perfil de Atraso de potência", )
ax.set_xlabel(r"Domínio de Atraso - $\tau\ \text{[}\mu$s]")
ax.set_ylabel("PDP")

cluster_power = results_1_1.clusters_power
azim = results_1_1.clusters_azimuth


print("Plotting power components in azimuth")
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='polar')
for i in range(len(azim)):
    ax.plot([azim[i], azim[i]], [1e-9, cluster_power[i]], color="black", linewidth=1)
markerline, *_ = ax.stem(azim, cluster_power, markerfmt="o", linefmt="k")
markerline.set(markerfacecolor='none')
ax.set_rscale("log")
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

fig.suptitle("Espectros de potência angulares (azimute)")

ax = fig.add_subplot(1, 2, 2)
ax.stem(np.rad2deg(azim), cluster_power, markerfmt="^k", linefmt="k")
ax.set_yscale("log")
ax.grid(True)

ax.set_xlabel(r"Azimute [$^\circ$]")
ax.set_ylabel(r"Potência")

# ax.set_title("Componentes com seus azimutes de chegada")

cluster_power = results_1_1.clusters_power
elev = results_1_1.clusters_elevation
print("Plotting power components in elevation")
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='polar')
for i in range(len(elev)):
    ax.plot([elev[i], elev[i]], [1e-9, cluster_power[i]], color="black", linewidth=1)
markerline, *_ = ax.stem(elev, cluster_power, markerfmt="o", linefmt="k")
markerline.set(markerfacecolor='none')
ax.set_rscale("log")
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

fig.suptitle("Espectros de potência angulares (elevação)")

ax = fig.add_subplot(1, 2, 2, )
ax.stem(np.rad2deg(elev), cluster_power, markerfmt="^k", linefmt="k")
ax.set_yscale("log")
ax.grid(True)

ax.set_xlabel(r"Elevação [$^\circ$]")
ax.set_ylabel(r"Potência")
# ax.set_title("Componentes com suas elevações de chegada", )


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

arrival_x = np.sin(elev) * np.cos(azim) * cluster_power
arrival_y = np.sin(elev) * np.sin(azim) * cluster_power
arrival_z = np.cos(elev) * cluster_power

ax.quiver(
    0, 0, 0,
    arrival_x, arrival_y, arrival_z,
    arrow_length_ratio=0.1
)

mx_p = np.max(cluster_power)

ax.set_xlim([-mx_p, mx_p])
ax.set_ylim([-mx_p, mx_p])
ax.set_zlim([-mx_p, mx_p])

ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.set_title('Vetores direção das componentes multipercurso')

doppler = results_1_1.clusters_doppler
print("Plotting power components in doppler domain")
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.stem(doppler, cluster_power, markerfmt="^k", linefmt="k")
ax.stem([doppler[0]], [cluster_power[0]], markerfmt="^b", linefmt="b")
ax.set_yscale("log")

ax.set_title(
    "$v = 5$ m/s"
)
ax.set_ylabel("Espectro Doppler")
ax.set_xlabel(r"Desvio Doppler - $\nu$ [Hz]")

ax = fig.add_subplot(1, 2, 2)
ax.stem(results_1_2.clusters_doppler, results_1_2.clusters_power, markerfmt="^k", linefmt="k")
ax.stem([results_1_2.clusters_doppler[0]], [results_1_2.clusters_power[0]], markerfmt="^b", linefmt="b")
ax.set_yscale("log")
ax.set_title(
    "$v = 50$ m/s"
)

print("Plotting received signal")
fig = plt.figure(
    # figsize=(12, 5)
)
ax = fig.add_subplot(111)
# ax = fig.add_subplot(1,2,1)
ax.plot(results_1_1.t_signal, np.abs(results_1_1.tx_signal), color="blue", label="Sinal Transmitido")
ax.plot(results_1_1.t_signal, np.abs(results_1_1.rx_signal), color="red", label="Sinal Recebido")
ax.legend()
ax.set_xlim((results_1_1.t_signal[0], results_1_1.t_signal[-1]))
ax.set_ylabel(r"$|\tilde{r}\ (t)|$")
ax.set_xlabel(r"Tempo [s]")
# ax = fig.add_subplot(1, 2, 2)
# ax.plot(results_1_2.t_signal, np.abs(results_1_2.tx_signal), color="blue", label="Sinal Transmitido")
# ax.plot(results_1_2.t_signal, np.abs(results_1_2.rx_signal), color="red", label="Sinal Recebido")

fig = plt.figure(
    # figsize=(12, 5)
)
ax = fig.add_subplot(111)
# ax = fig.add_subplot(1, 2, 1)
ax.plot(results_2_1.t_signal, np.abs(results_2_1.tx_signal), color="blue", label="Sinal Transmitido")
ax.plot(results_2_1.t_signal, np.abs(results_2_1.rx_signal), color="red", label="Sinal Recebido")
ax.set_ylabel(r"$|\tilde{r}\ (t)|$")
ax.set_xlabel(r"Tempo [s]")
# ax = fig.add_subplot(1, 2, 2)
# ax.plot(results_2_2.t_signal, np.abs(results_2_2.tx_signal), color="blue", label="Sinal Transmitido")
# ax.plot(results_2_2.t_signal, np.abs(results_2_2.rx_signal), color="red", label="Sinal Recebido")
ax.set_xlim((results_2_1.t_signal[0], results_2_1.t_signal[-1]))
ax.legend()

fig = plt.figure(
    # figsize=(12, 5)
)
# ax = fig.add_subplot(1, 2, 1)
ax = fig.add_subplot(111)
# until = np.argmax(results_3_1.t_signal > )
ax.plot(results_3_1.t_signal, np.abs(results_3_1.tx_signal), color="blue", label="Sinal Transmitido")
ax.plot(results_3_1.t_signal, np.abs(results_3_1.rx_signal), color="red", label="Sinal Recebido")
ax.set_ylabel(r"$|\tilde{r}\ (t)|$")
ax.set_xlabel(r"Tempo [s]")
ax.set_xlim((results_3_1.t_signal[0], 5*1e-7))
# ax = fig.add_subplot(1, 2, 2)
# ax.plot(results_3_2.t_signal, np.abs(results_3_2.tx_signal), color="blue", label="Sinal Transmitido")
# ax.plot(results_3_2.t_signal, np.abs(results_3_2.rx_signal), color="red", label="Sinal Recebido")
ax.legend()
# plt.show()

def build_autocorr(alpha_sqrd, tau, ni):
    Ohmega_c = np.sum(alpha_sqrd)

    def autocorr(kappa: np.ndarray, ohmega: np.ndarray):
        kappa = np.reshape(kappa, (1, len(kappa)))
        ohmega = np.reshape(ohmega, (1, len(ohmega)))

        return np.sum(
            alpha_sqrd[:, np.newaxis]
                * np.exp(-2j * np.pi * tau[: ,np.newaxis] * kappa)
                * np.exp(2j * np.pi
                         * ni[:, np.newaxis]
                         * ohmega
                     )
            ,axis=0
        ) / Ohmega_c

    return autocorr

s = 0 if scenario.los else 1

chosen_results = results_3_2
autocorr_1_1 = build_autocorr(
    chosen_results.clusters_power[s:],
    chosen_results.clusters_delay[s:],
    chosen_results.clusters_doppler[s:]
)
max_vn = np.max(chosen_results.clusters_doppler[0 if scenario.los else 1:])
ds = chosen_results.delay_spread*1e6
speed = 50

rho_t_1 = 0.95
rho_t_2 = 0.9

x = np.pow(10, np.linspace(-60, 0, 10000) / 10)
y = np.abs(autocorr_1_1(np.zeros_like(x), x))

sigma_at_rho_t_1 = np.argmax(y <= rho_t_1)
sigma_at_rho_t_2 = np.argmax(y <= rho_t_2)

# fig, ax = plt.subplots()
fig = plt.figure(
    figsize=(12, 5)
)
ax = fig.add_subplot(111)
ax.plot(x, np.abs(y), color="blue")
ax.set_title(
    rf"$T_C(\rho_T = {rho_t_1}) = {x[sigma_at_rho_t_1]*1e3:.2g}$ ms, "
    rf"$T_C(\rho_T = {rho_t_2}) = {x[sigma_at_rho_t_2]*1e3:.2g}$ ms, "
    rf"max$(\nu_n) = {max_vn:.0f}$ Hz, "
    rf"$v_{{rx}} = {speed}$ m/s"
)
ax.set_xscale("log")
ax.xaxis.set_label_text(r"Desvio de Tempo - $\sigma$ (s)")
ax.vlines([x[sigma_at_rho_t_1], x[sigma_at_rho_t_2]], -0.1, 1.1, linestyle=":", color="k")
ax.hlines([np.abs(y[sigma_at_rho_t_1]), np.abs(y[sigma_at_rho_t_2])], x[0], x[-1], linestyle=":", color="k")
ax.set_ylim((-0.1, 1.1))
ax.set_xlim((1e-6, 1))
# ax_mean.set_xlim((0.5, 100))
ax.yaxis.set_label_text(r"|$\rho_{TT}(0,\sigma)$|")
# fig.show()
# plt.show()

x = np.pow(10, np.linspace(0, 100, 10000) / 10)
y = np.abs(autocorr_1_1(x, np.zeros_like(x)))

sigma_at_rho_t_1 = np.where(y <= rho_t_1)[0][0]-1
sigma_at_rho_t_2 = np.where(y <= rho_t_2)[0][0]-1

# fig, ax = plt.subplots()
fig = plt.figure(
    figsize=(12, 5)
)
ax = fig.add_subplot(111)
# ax = fig.add_subplot(1, 2, 2)
ax.plot(x, np.abs(y), color="blue")
# print("results.delay_spread", results.delay_spread)
# exit()
ax.set_title(
    rf"$B_C(\rho_B = {rho_t_1}) = {x[sigma_at_rho_t_1]/1e6:.2g}$ MHz, "
    rf"$B_C(\rho_B = {rho_t_2}) = {x[sigma_at_rho_t_2]/1e6:.2g}$ MHz, "
    rf"$\rho_{{\tau}} = {ds:.2g}\, \mu$s"
)
ax.set_xscale("log")
ax.xaxis.set_label_text(r"Desvio de Frequência - $\kappa$ (Hz)")
ax.vlines([x[sigma_at_rho_t_1], x[sigma_at_rho_t_2]], -0.1, 1.1, linestyle=":", color="k")
ax.hlines([np.abs(y[sigma_at_rho_t_1]), np.abs(y[sigma_at_rho_t_2])], x[0], x[-1], linestyle=":", color="k")
ax.set_ylim((-0.1, 1.1))
ax.set_xlim((1, 1e10))
# ax.yaxis.set_label_text(r"|$\rho_{TT}(\kappa,0)$|")

plt.show()
