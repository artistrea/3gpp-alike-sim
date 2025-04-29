from dataclasses import dataclass
from enum import Enum
import numpy as np


class PropagationModelEnum(Enum):
    UMi = 1
    UMa = 2


@dataclass
class Vec3Params():
    x: float
    y: float
    z: float

    @staticmethod
    def make_from(val) -> "Vec3Params":
        if (isinstance(val, tuple) or isinstance(val, list)) and len(val) == 3:
            return Vec3Params(*val)
        if isinstance(val, Vec3Params):
            return Vec3Params(val.x, val.y, val.z)

        raise ValueError(f"Cannot convert '{val}' into Vec3Params")

    @property
    def norm(self):
        return np.linalg.norm([self.x, self.y, self.z])


@dataclass
class ScenarioParams():
    seed: int

    f_c: float
    ut_velocity: Vec3Params
    ut_position: Vec3Params
    bs_position: Vec3Params
    # TODO: different parametrization, and set diff. los for each link
    los: bool
    # TODO: set propagation model for each link
    propagation_model: PropagationModelEnum

    def __post_init__(self):
        self.ut_position = Vec3Params.make_from(self.ut_position)
        self.ut_velocity = Vec3Params.make_from(self.ut_velocity)
        self.bs_position = Vec3Params.make_from(self.bs_position)

    def get_shadowing_std(
        self,
    ) -> float:
        return self._get_shadowing_std(
            self.propagation_model,
            self.los
        )

    def get_aoa_azim_std_stats(
        self,
    ) -> (float, float):
        return self._get_aoa_azim_stats(
            self.f_c,
            self.propagation_model,
            self.los
        )

    # AOA is ZOA in 3gpp docs
    def get_aoa_elev_std_stats(
        self,
    ) -> (float, float):
        return self._get_aoa_elev_stats(
            self.f_c,
            self.propagation_model,
            self.los
        )

    def get_rice_factor_std(
        self,
    ) -> float:
        return self._get_rice_factor_std(
            self.propagation_model,
            self.los
        )

    
    def get_delay_spread_stats(
        self,
    ) -> (float, float):
        return self._get_delay_spread_stats(
            self.f_c,
            self.propagation_model,
            self.los
        )

    def get_delay_proportionality_factor(
        self,
    ) -> float:
        return self._get_delay_proportionality_factor(
            self.propagation_model,
            self.los
        )

    @staticmethod
    def _get_aoa_azim_stats(
        f_c: float,
        propagation_model: PropagationModelEnum,
        los: bool
    ) -> (float, float):
        """
        Returns (mean, std_variance)
        """
        if propagation_model is PropagationModelEnum.UMi:
            if los:
                mean = -0.08 * np.log10(1 + f_c) + 1.73
                std = 0.14 * np.log10(1 + f_c) + 0.28
            else:
                mean = -0.08 * np.log10(1 + f_c) + 1.81
                std = 0.05 * np.log10(1 + f_c) + 0.3
        elif propagation_model is PropagationModelEnum.UMa:
            if los:
                mean = 1.81
                std = 0.2
            else:
                mean = -0.27 * np.log10(f_c) + 2.08
                std = 0.11
        else:
            raise ValueError(f"propagation_model = {propagation_model} não dá")

        if isinstance(f_c, np.ndarray):
            std = np.zeros_like(f_c) + std
            mean = np.zeros_like(f_c) + mean

        return (mean, std)

    @staticmethod
    def _get_aoa_elev_stats(
        f_c: float,
        propagation_model: PropagationModelEnum,
        los: bool
    ) -> (float, float):
        """
        Returns (mean, std_variance)
        """
        if propagation_model is PropagationModelEnum.UMi:
            if los:
                mean = -0.1 * np.log10(1 + f_c) + 0.73
                std = -0.07 * np.log10(1 + f_c) + 0.34
            else:
                mean = -0.04 * np.log10(1 + f_c) + 0.92
                std = -0.07 * np.log10(1 + f_c) + 0.41
        elif propagation_model is PropagationModelEnum.UMa:
            if los:
                mean = 0.95
                std = 0.16
            else:
                mean = -0.3236 * np.log10(f_c) + 1.512
                std = 0.16
        else:
            raise ValueError(f"propagation_model = {propagation_model} não dá")
        if isinstance(f_c, np.ndarray):
            std = np.zeros_like(f_c) + std
            mean = np.zeros_like(f_c) + mean

        return (mean, std)

    @staticmethod
    def _get_rice_factor_std(
        propagation_model: PropagationModelEnum,
        los: bool
    ) -> float:
        if propagation_model is PropagationModelEnum.UMi:
            if los:
                std = 5
            else:
                std = 0
        elif propagation_model is PropagationModelEnum.UMa:
            if los:
                std = 3.5
            else:
                std = 0
        else:
            raise ValueError(f"propagation_model = {propagation_model} não dá")

        return std
    @staticmethod
    def _get_shadowing_std(
        propagation_model: PropagationModelEnum,
        los: bool
    ) -> float:
        if propagation_model is PropagationModelEnum.UMi:
            if los:
                std = 4
            else:
                std = 7.82
        elif propagation_model is PropagationModelEnum.UMa:
            if los:
                std = 4
            else:
                std = 6
        else:
            raise ValueError(f"propagation_model = {propagation_model} não dá")

        return std

    @staticmethod
    def _get_delay_proportionality_factor(
        propagation_model: PropagationModelEnum,
        los: bool
    ) -> float:
        if propagation_model is PropagationModelEnum.UMi:
            if los:
                factor = 3
            else:
                factor = 2.1
        elif propagation_model is PropagationModelEnum.UMa:
            if los:
                factor = 2.5
            else:
                factor = 2.3
        else:
            raise ValueError(f"propagation_model = {propagation_model} não dá")

        return factor

    @staticmethod
    def _get_delay_spread_stats(
        f_c: float,
        propagation_model: PropagationModelEnum,
        los: bool
    ) -> (float, float):
        """
        Returns (mean, std_variance)
        """
        if propagation_model is PropagationModelEnum.UMi:
            if los:
                mean = -0.24 * np.log10(1 + f_c) - 7.14
                std = 0.38
            else:
                mean = -0.24 * np.log10(1 + f_c) - 6.83
                std = -0.16 * np.log10(1 + f_c) + 0.28
        elif propagation_model is PropagationModelEnum.UMa:
            if los:
                mean = -0.0963 * np.log10(1 + f_c) - 6.955
                std = 0.66
            else:
                mean = -0.204 * np.log10(1 + f_c) - 6.28
                std = 0.39
        else:
            raise ValueError(f"propagation_model = {propagation_model} não dá")
        if isinstance(f_c, np.ndarray):
            std = np.zeros_like(f_c) + std
        return (mean, std)
