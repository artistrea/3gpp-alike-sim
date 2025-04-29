import numpy as np
from simulator.parameters import ScenarioParams
from simulator.results import ScenarioResults
import matplotlib.pyplot as plt

def run_simulation(
    scenario: ScenarioParams,
    tx_signal: np.array,
    tx_time: np.array
) -> ScenarioResults:
    assert(tx_signal.shape == tx_time.shape)

    results = ScenarioResults()
    results.t_signal = tx_time
    results.tx_signal = tx_signal

    rng = np.random.default_rng(scenario.seed)

    N_PATHS = 100

    ###########################################################################
    # GET DELAYS
    delay_spread_mean, delay_spread_std = scenario.get_delay_spread_stats()
    sigma_tau_log10 = rng.normal(delay_spread_mean, delay_spread_std)
    sigma_tau = np.power(10, sigma_tau_log10)

    results.delay_spread = sigma_tau

    r_tau = scenario.get_delay_proportionality_factor()

    # delay samples
    tau_p = rng.exponential(r_tau * sigma_tau, N_PATHS)
    # normalized delay samples:
    tau = np.sort(
        tau_p - np.min(
                tau_p,
            ),
    )
    results.clusters_delay = tau

    ###########################################################################
    # GET CLUSTERS POWERS
    Kr_dB = rng.normal(0, scenario.get_rice_factor_std())
    if scenario.los:
        K = np.power(10, Kr_dB/10)
    else:
        K = 0
    results.rice_factor = K

    # multipath power
    ksi = rng.normal(
        0, scenario.get_shadowing_std(), N_PATHS
    )

    cluster_power = np.exp(-tau * (r_tau - 1)/ (r_tau * sigma_tau)) * np.power(10, -ksi / 10)

    # NOTE: not according to original document
    cluster_power[1:] = cluster_power[1:] / np.sum(cluster_power[1:]) / (K+1)
    cluster_power[0] = K / (K + 1)

    results.clusters_power = cluster_power

    ###########################################################################
    # GET ARRIVAL AZIMUTH
    azim_std_mean, azim_std_std = scenario.get_aoa_azim_std_stats()
    azim_std_log = rng.normal(azim_std_mean, azim_std_std)
    azim_std = np.deg2rad(np.pow(10, azim_std_log))

    azim_p = 1.42 * azim_std * np.sqrt(
            -np.log(cluster_power / np.max(cluster_power))
        )
    U = rng.choice([-1,1], size=N_PATHS)
    Y = rng.normal(0, azim_std/7, size=N_PATHS)
    azim = U * azim_p + Y

    if scenario.los:
        # if los, highest power cluster is azim = 0
        # since in local coords, the two equipments are in the same x axis
        azim -= azim[0]

    results.clusters_azimuth = azim

    ###########################################################################
    # GET ARRIVAL ELEVATION
    elev_std_mean, elev_std_std = scenario.get_aoa_elev_std_stats()
    elev_std_log = rng.normal(elev_std_mean, elev_std_std)
    elev_std = np.deg2rad(np.pow(10, elev_std_log))
    elev_p = -elev_std * np.log(cluster_power / np.max(cluster_power))

    # NOTE: professor asked to arbitrarily choose some mean
    # so i chose the los path between both stations to always be the mean
    elev_mean = np.atan2(
        np.sqrt(
            (scenario.bs_position.y - scenario.ut_position.y) ** 2
            + (scenario.bs_position.x - scenario.ut_position.x) ** 2
        ),
        scenario.bs_position.z - scenario.ut_position.z
    )

    U = rng.choice([-1,1], size=N_PATHS)
    # check if actually need to convert azim_std to deg before this
    Y = rng.normal(0, elev_std/7, size=N_PATHS)
    elev = U * elev_p + Y
    if scenario.los:
        elev = elev - elev[0] + elev_mean

    results.clusters_elevation = elev

    # # elevation angles should be in [0, pi] rad
    # elev = np.abs(elev)
    # elev[elev > np.pi] = 2 * np.pi - elev[elev > np.pi]

    ###########################################################################
    # GET DOPPLER
    arrival_x = np.sin(elev) * np.cos(azim)
    arrival_y = np.sin(elev) * np.sin(azim)
    arrival_z = np.cos(elev)

    speed = np.array([[scenario.ut_velocity.x, scenario.ut_velocity.y, scenario.ut_velocity.z]])

    arrival_vecs = np.transpose([arrival_x, arrival_y, arrival_z])

    lmbda = 3e8 / (scenario.f_c * 1e9)
    doppler = np.vecdot(arrival_vecs, speed) / lmbda
    results.clusters_doppler = doppler

    ###########################################################################
    # GET RECEIVED SIGNAL
    rx = np.zeros((N_PATHS, len(tx_time)), dtype=np.complex128)
    step = tx_time[1] - tx_time[0]
    tau_i = np.floor(tau / step).astype(int)

    # NOTE: we need to do this because we destroy the first cluster in nlos case
    s = 0 if scenario.los else 1

    for i in range(s, N_PATHS):
        rx[i, tau_i[i]:] = np.sqrt(
            cluster_power[i]
        ).astype(np.complex128) * np.exp(
            -2j * np.pi * (
                (scenario.f_c * 1e9 + doppler[i]) * tau[i]
                - doppler[i] * tx_time[tau_i[i]:]
            )
        ) * tx_signal[:len(tx_time)-tau_i[i]].astype(np.complex128)

    rx = np.sum(rx, axis=0)

    results.rx_signal = rx

    return results

