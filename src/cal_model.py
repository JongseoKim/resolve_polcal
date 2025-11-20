# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2023 Max-Planck-Society

import numpy as np
import resolve as rve #TODO import specific functions in resolve
import nifty8 as ift
import ducc0

from astropy.coordinates import EarthLocation
import astropy.time as at
import datetime as dt

from .ift_cfm_maker import cfm_from_cfg
from .utilities import Expander_00DD, Expander_0D0D, Expander_D0D0, Expander_DD00, Expander_NNPP, Expander_PNPN


def dofdex_or_none(cfg, key, total_N):
    if cfg[key] == "False":
        return None
    if cfg[key] == "True":
        return np.arange(total_N)
    else:
        return np.fromstring(cfg[key], dtype=int, sep=',')


def built_cal_model(config, obs, single_gain=False, tmin=None, tmax=None, gain_phase_config_name ="gain_phase", gain_logamplitude_config_name="gain_logamplitude"):
    nthreads = config["base"].getint("nthreads_rve")
    solint = config[gain_phase_config_name].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    if tmin==None and tmax==None:
        tmin, tmax = rve.tmin_tmax(obs)
    #assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    total_N = 2 * len(uantennas)
    if single_gain==True:
        total_N = len(uantennas)
    dofdex_phase = dofdex_or_none(config[gain_phase_config_name], f"diff_correlation_kernels", total_N)
    dofdex_logamp = dofdex_or_none(config[gain_logamplitude_config_name], f"diff_correlation_kernels", total_N)

    dd = {"time": time_domain}
    cfm_kwargs_phase = {"total_N": total_N, "dofdex":dofdex_phase}
    cfm_kwargs_logamp = {"total_N": total_N, "dofdex":dofdex_logamp}
    phase_ = cfm_from_cfg(
        config[gain_phase_config_name],
        dd,
        gain_phase_config_name,
        domain_prefix=gain_phase_config_name,
        **cfm_kwargs_phase,
    ).finalize(0)

    uncorrelated_gain_phase = config[gain_phase_config_name]["uncorrelated_gain_phase"]
    phase_amp = config[gain_phase_config_name].getfloat("uncorrelated_gain_phase_amp")

    if uncorrelated_gain_phase == "True":
        phase_ = phase_amp * np.pi * ift.FieldAdapter(phase_.target, gain_phase_config_name)

    logamp_ = cfm_from_cfg(
        config[gain_logamplitude_config_name],
        dd,
        gain_logamplitude_config_name,
        domain_prefix=gain_logamplitude_config_name,
        **cfm_kwargs_logamp,
    ).finalize(0)

    if single_gain == False:
        obs_stokes_i = obs.restrict_to_stokesi()
    else:
        obs_stokes_i = obs
    pdom, _, fdom = obs_stokes_i.vis.domain
    reshaper = ift.DomainChangerAndReshaper(
        phase_.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )


    phase = reshaper @ phase_
    logamp = reshaper @ logamp_
    calibration_operator = rve.calibration_distribution(
        obs, phase, logamp, antenna_dct, None
    )


    model_dict = {}
    model_dict["gain_phase"] = phase
    model_dict["gain_logamp"] = logamp
    model_dict["calibration_operator"] = calibration_operator

    return model_dict


def amp_calibration_op_uncorrelated(obs, std):
    utimes = rve.unique_times(obs)
    uants = obs.antenna_positions.unique_antennas()
    dom = [obs.polarization.space] + [
        ift.UnstructuredDomain(nn) for nn in [len(uants), len(utimes), obs.nfreq]
    ]
    time_dct = {aa: ii for ii, aa in enumerate(utimes)}
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}
    inp = ift.ScalingOperator(dom, 1.0).ducktape(f"amplitude calib xi ")

    cop1 = rve.CalibrationDistributor(
        dom, obs.vis.domain, obs.ant1, obs.time, antenna_dct, time_dct
    )
    cop2 = rve.CalibrationDistributor(
        dom, obs.vis.domain, obs.ant2, obs.time, antenna_dct, time_dct
    )
    cop = (cop1 + cop2) @ (std * inp)
    cop = cop.ducktape_left(f"cal")

    return cop


def amp_calibration_op_correlated(config, obs, gain_config_name="gain_logamplitude"):
    solint = 10
    zero_padding_factor = 2

    utimes = rve.unique_times(obs)
    uants = obs.antenna_positions.unique_antennas()
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}

    tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    total_N = len(uants)
    dofdex_logamp = dofdex_or_none(config[gain_config_name], f"diff_correlation_kernels", total_N)

    dd = {"time": time_domain}
    cfm_kwargs_logamp = {"total_N": total_N, "dofdex": dofdex_logamp}

    logamp_ = cfm_from_cfg(
        config[gain_config_name],
        dd,
        gain_config_name,
        domain_prefix=gain_config_name,
        **cfm_kwargs_logamp,
    ).finalize(0)

    obs_stokes_i = obs.restrict_to_stokesi()
    pdom, _, fdom = obs_stokes_i.vis.domain
    reshaper = ift.DomainChangerAndReshaper(
        logamp_.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uants)),
            time_domain,
            fdom
        )
    )
    logamp = reshaper @ logamp_

    cop1 = rve.CalibrationDistributor(logamp.target, obs.vis.domain, obs.ant1, obs.time, antenna_dct, None)
    cop2 = rve.CalibrationDistributor(logamp.target, obs.vis.domain, obs.ant2, obs.time, antenna_dct, None)

    cop = (cop1 + cop2) @ logamp
    cop = cop.ducktape_left(f"cal")

    return cop, logamp


def gain_ops(config, obs, gaincal=True, tmin=None, tmax=None):
    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2
    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    nfreq = obs.nfreq
    if tmin==None and tmax==None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    total_N = len(uantennas) * nfreq
    dofdex_phase = dofdex_or_none(config["gain_phase"], f"diff_correlation_kernels", total_N)
    dofdex_logamp = dofdex_or_none(config["gain_logamplitude"], f"diff_correlation_kernels", total_N)

    dd = {"time": time_domain}
    cfm_kwargs_phase = {"total_N": total_N, "dofdex":dofdex_phase}
    cfm_kwargs_logamp = {"total_N": total_N, "dofdex":dofdex_logamp}
    phase_RCP_ = cfm_from_cfg(
        config["gain_phase"],
        dd,
        "gain_phase",
        domain_prefix="gain_phase_RCP_",
        **cfm_kwargs_phase,
    ).finalize(0)
    phase_LCP_ = cfm_from_cfg(
        config["gain_phase"],
        dd,
        "gain_phase",
        domain_prefix="gain_phase_LCP_",
        **cfm_kwargs_phase,
    ).finalize(0)

    uncorrelated_gain_phase = config["gain_phase"]["uncorrelated_gain_phase"]
    phase_amp = config["gain_phase"].getfloat("uncorrelated_gain_phase_amp")

    if uncorrelated_gain_phase == "True":
        phase_RCP_ = phase_amp * np.pi * ift.FieldAdapter(phase_RCP_.target, "gain_phase_RCP_")
        phase_LCP_ = phase_amp * np.pi * ift.FieldAdapter(phase_LCP_.target, "gain_phase_LCP_")

    logamp_RCP_ = cfm_from_cfg(
        config["gain_logamplitude"],
        dd,
        "gain_logamplitude",
        domain_prefix="gain_logamplitude_RCP_",
        **cfm_kwargs_logamp,
    ).finalize(0)

    logamp_LCP_ = cfm_from_cfg(
        config["gain_logamplitude"],
        dd,
        "gain_logamplitude",
        domain_prefix="gain_logamplitude_LCP_",
        **cfm_kwargs_logamp,
    ).finalize(0)

    if gaincal == False:
        logamp_RCP_ = ift.ScalingOperator(logamp_RCP_.target, 0) @ logamp_RCP_
        logamp_LCP_ = ift.ScalingOperator(logamp_LCP_.target, 0) @ logamp_LCP_
        phase_RCP_ = ift.ScalingOperator(phase_RCP_.target, 0) @ phase_RCP_
        phase_LCP_ = ift.ScalingOperator(phase_LCP_.target, 0) @ phase_LCP_

    return logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_

def Dterm_ops(config, obs):
    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    total_N = 2 * len(uantennas)
    dofdex_Dterm_phase = dofdex_or_none(config["Dterm_phase"], f"diff_correlation_kernels", total_N)
    dofdex_Dterm_logamp = dofdex_or_none(config["Dterm_logamplitude"], f"diff_correlation_kernels", total_N)

    dd = {"time": time_domain}
    cfm_kwargs_phase = {"total_N": total_N, "dofdex":dofdex_Dterm_phase}
    cfm_kwargs_logamp = {"total_N": total_N, "dofdex":dofdex_Dterm_logamp}
    Dterm_phase_ = cfm_from_cfg(
        config["Dterm_phase"],
        dd,
        "Dterm_phase",
        domain_prefix="Dterm_phase",
        **cfm_kwargs_phase,
    ).finalize(0)

    Dterm_logamp_ = cfm_from_cfg(
        config["Dterm_logamplitude"],
        dd,
        "Dterm_logamplitude",
        domain_prefix="Dterm_logamplitude",
        **cfm_kwargs_logamp,
    ).finalize(0)

    return Dterm_logamp_, Dterm_phase_


def Const_Dterm_ops(config, obs, tmin=None, tmax=None):
    solint = config["gain_phase"].getint("solution_interval")
    const_Dterm_logamp_mean = config["Dterm_logamplitude"].getfloat("const_Dterm_logamp_mean")
    const_Dterm_logamp_std = config["Dterm_logamplitude"].getfloat("const_Dterm_logamp_std")
    const_Dterm_phase_mean = config["Dterm_phase"].getfloat("const_Dterm_phase_mean")
    const_Dterm_phase_std = config["Dterm_phase"].getfloat("const_Dterm_phase_std")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    nfreq = obs.nfreq
    if tmin==None and tmax==None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    total_N = len(uantennas) * nfreq

    domain = ift.DomainTuple.make([ift.UnstructuredDomain(total_N)])
    target = ift.DomainTuple.make([ift.UnstructuredDomain(len(uantennas)), time_domain, ift.UnstructuredDomain(nfreq)])

    Const_Dterm1_RCP = ift.NormalTransform(const_Dterm_logamp_mean, const_Dterm_logamp_std, "Dterm_logamp_RCP", N_copies=total_N)
    Const_Dterm1_LCP = ift.NormalTransform(const_Dterm_logamp_mean, const_Dterm_logamp_std, "Dterm_logamp_LCP", N_copies=total_N)
    Const_Dterm2_RCP = ift.NormalTransform(const_Dterm_phase_mean, const_Dterm_phase_std, "Dterm_phase_RCP", N_copies=total_N)
    Const_Dterm2_LCP = ift.NormalTransform(const_Dterm_phase_mean, const_Dterm_phase_std, "Dterm_phase_LCP", N_copies=total_N)

    reshaper = ift.DomainChangerAndReshaper(
        [ift.UnstructuredDomain(total_N)],
        [
            ift.UnstructuredDomain(len(uantennas)),
            ift.UnstructuredDomain(nfreq)
        ]
    )

    Const_Dterm1_RCP = reshaper @ Const_Dterm1_RCP
    Const_Dterm1_LCP = reshaper @ Const_Dterm1_LCP
    Const_Dterm2_RCP = reshaper @ Const_Dterm2_RCP
    Const_Dterm2_LCP = reshaper @ Const_Dterm2_LCP

    Expander_Op = ift.ContractionOperator(target, 1).adjoint
    Const_Dterm_logamp_RCP_ = Expander_Op @ Const_Dterm1_RCP
    Const_Dterm_logamp_LCP_ = Expander_Op @ Const_Dterm1_LCP
    Const_Dterm_phase_RCP_ = Expander_Op @ Const_Dterm2_RCP
    Const_Dterm_phase_LCP_ = Expander_Op @ Const_Dterm2_LCP

    return Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_


def Const_Dterm_ops_normal(config, obs):
    # Dterm = a+bi
    solint = config["gain_phase"].getint("solution_interval")
    const_Dterm_a_mean = config["Dterm_logamplitude"].getfloat("const_Dterm_a_mean")
    const_Dterm_a_std = config["Dterm_logamplitude"].getfloat("const_Dterm_a_std")
    const_Dterm_b_mean = config["Dterm_logamplitude"].getfloat("const_Dterm_b_mean")
    const_Dterm_b_std = config["Dterm_logamplitude"].getfloat("const_Dterm_b_std")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    total_N = len(uantennas)

    domain = ift.DomainTuple.make([ift.UnstructuredDomain(total_N)])
    target = ift.DomainTuple.make([ift.UnstructuredDomain(total_N), time_domain])

    Const_Dterm_a_RCP = ift.NormalTransform(const_Dterm_a_mean, const_Dterm_a_std, "Dterm_a_RCP", N_copies=total_N)
    Const_Dterm_a_LCP = ift.NormalTransform(const_Dterm_a_mean, const_Dterm_a_std, "Dterm_a_LCP", N_copies=total_N)
    Const_Dterm_b_RCP = ift.NormalTransform(const_Dterm_b_mean, const_Dterm_b_std, "Dterm_b_RCP", N_copies=total_N)
    Const_Dterm_b_LCP = ift.NormalTransform(const_Dterm_b_mean, const_Dterm_b_std, "Dterm_b_LCP", N_copies=total_N)

    Expander_Op = ift.ContractionOperator(target, 1).adjoint

    Const_Dterm_a_RCP_ = Expander_Op @ Const_Dterm_a_RCP
    Const_Dterm_a_LCP_ = Expander_Op @ Const_Dterm_a_LCP
    Const_Dterm_b_RCP_ = Expander_Op @ Const_Dterm_b_RCP
    Const_Dterm_b_LCP_ = Expander_Op @ Const_Dterm_b_LCP

    Const_Dterm_RCP = Const_Dterm_a_RCP_ + 1j * Const_Dterm_b_RCP_
    Const_Dterm_LCP = Const_Dterm_a_LCP_ + 1j * Const_Dterm_b_LCP_

    #Const_Dterm_logamp_RCP_ = ift.log(ift.abs(Const_Dterm_RCP))
    #Const_Dterm_logamp_LCP_ = ift.log(ift.abs(Const_Dterm_LCP))
    Const_Dterm_logamp_RCP_ = ift.log(ift.sqrt(Const_Dterm_a_RCP_.power(2) + Const_Dterm_b_RCP_.power(2)))
    Const_Dterm_logamp_LCP_ = ift.log(ift.sqrt(Const_Dterm_a_LCP_.power(2) + Const_Dterm_b_LCP_.power(2)))

    #TODO solve angle (use arctan2)
    Const_Dterm_phase_RCP_ = ift.arctan(Const_Dterm_b_RCP_ * Const_Dterm_a_RCP_.power(-1))
    Const_Dterm_phase_LCP_ = ift.arctan(Const_Dterm_b_LCP_ * Const_Dterm_a_LCP_.power(-1))

    return Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_



def get_field_rotation_angle_field(config, obs, BeginTime_UTC_list, timestamp, antenna_mount=None, tmin=None, tmax=None):
    rve.my_assert_isinstance(obs, rve.Observation)
    cfg_obs = config["observation"]
    if tmin==None and tmax==None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0

    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    if antenna_mount == None:
        antenna_mount_type = obs._auxiliary_tables["ANTENNA"]["MOUNT"]
    else:
        antenna_mount_type = antenna_mount
    station_names = obs._auxiliary_tables["ANTENNA"]["STATION"]
    nfreq = obs.nfreq

    time_array = np.arange(0, solint * ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint)
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )
    total_N = len(uantennas)
    domain = ift.DomainTuple.make((ift.UnstructuredDomain(total_N), time_domain))

    obs_direction = obs.direction.phase_center
    source_RA = np.rad2deg(obs_direction[0])
    source_DEC = np.rad2deg(obs_direction[1])

    timestamp_array = time_array + timestamp[0]
    yeararr = np.zeros(len(time_array), dtype='int')
    montharr = np.zeros(len(time_array), dtype='int')
    dayarr = np.zeros(len(time_array), dtype='int')
    hourarr = np.zeros(len(time_array), dtype='int')
    minutearr = np.zeros(len(time_array), dtype='int')
    secondarr = np.zeros(len(time_array), dtype='int')

    for ii in range(len(time_array)):
        yeararr[ii] = int(dt.datetime.utcfromtimestamp(timestamp_array[ii]).strftime("%Y"))
        montharr[ii] = int(dt.datetime.utcfromtimestamp(timestamp_array[ii]).strftime("%m"))
        dayarr[ii] = int(dt.datetime.utcfromtimestamp(timestamp_array[ii]).strftime("%d"))
        hourarr[ii] = int(dt.datetime.utcfromtimestamp(timestamp_array[ii]).strftime("%H"))
        minutearr[ii] = int(dt.datetime.utcfromtimestamp(timestamp_array[ii]).strftime("%M"))
        secondarr[ii] = float(dt.datetime.utcfromtimestamp(timestamp_array[ii]).strftime("%S.%f"))

    rve.my_assert(np.max(hourarr) < 24)
    rve.my_assert(np.max(minutearr) < 60)
    rve.my_assert(np.max(secondarr) < 60)

    dumdatetime = [dt.datetime(a, b, c, d, e, f) for a, b, c, d, e, f in
                   zip(yeararr, montharr, dayarr, hourarr, minutearr, secondarr)]

    dumt = []
    for dum in dumdatetime:
        dumt.append("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:f}".format(dum.year,
                                                                     dum.month, dum.day, dum.hour, dum.minute,
                                                                     dum.second + dum.microsecond * 1e-6))
    dumt = at.Time(dumt)
    gst = dumt.sidereal_time('mean', 'greenwich').hour # UT -> GST
    ant_coord = obs.antenna_coordinates

    station_long_arr = []
    station_lat_arr = []
    station_height_arr = []

    for ii in range(len(uantennas)):
        station_long_arr.append(
            EarthLocation.from_geocentric(ant_coord[list(uantennas)[ii], 0], ant_coord[list(uantennas)[ii], 1], ant_coord[list(uantennas)[ii], 2], unit='m').to_geodetic()[0].value)  # unit degree
        station_lat_arr.append(
            EarthLocation.from_geocentric(ant_coord[list(uantennas)[ii], 0], ant_coord[list(uantennas)[ii], 1], ant_coord[list(uantennas)[ii], 2], unit='m').to_geodetic()[1].value)  # unit degree
        station_height_arr.append(
            EarthLocation.from_geocentric(ant_coord[list(uantennas)[ii], 0], ant_coord[list(uantennas)[ii], 1], ant_coord[list(uantennas)[ii], 2], unit='m').to_geodetic()[2].value)

    hangle_arr = []
    parang_arr = []
    altitude_arr = []

    for ii in range(len(uantennas)):
        hangle_arr.append(
            np.radians(gst * 15. + station_long_arr[ii] - source_RA))  # hour angle = GST + station_long - source_RA, [rad]
    for ii in range(len(uantennas)):
        parang_arr.append(np.arctan2((np.sin(hangle_arr[ii]) * np.cos(np.radians(station_lat_arr[ii]))),
                                     (np.sin(np.radians(station_lat_arr[ii]) * np.cos(np.radians(source_DEC)) - np.cos(
                                         np.radians(station_lat_arr[ii])) * np.sin(np.radians(source_DEC)) * np.cos(
                                         hangle_arr[ii])))))
    for ii in range(len(uantennas)):
        altitude_arr.append(np.arcsin(np.sin(np.radians(source_DEC)) * np.sin(np.radians(station_lat_arr[ii])) + np.cos(
            np.radians(source_DEC)) * np.cos(np.radians(station_lat_arr[ii])) * np.cos(hangle_arr[ii])))

    f_el_arr = np.zeros(len(uantennas))  #TODO get f_el_arr from obs
    f_par_arr = np.zeros(len(uantennas))
    phi_off_arr = np.zeros(len(uantennas)) #TODO get phi_off_arr from obs

    for ii in range(len(uantennas)):
        if antenna_mount_type[list(uantennas)[ii]] == 'ALT-AZ':
            f_par_arr[ii] = 1
            phi_off_arr[ii] = 0
        elif antenna_mount_type[list(uantennas)[ii]] == 'ALT-AZ+NASMYTH-R':

            f_el_arr[ii] = 1
            f_par_arr[ii] = 1
            phi_off_arr[ii] = 0
        elif antenna_mount_type[list(uantennas)[ii]] == 'ALT-AZ+NASMYTH-L':
            f_el_arr[ii] = -1
            f_par_arr[ii] = 1
            if station_names[list(uantennas)[ii]] == "SW":
                phi_off_arr[ii] = np.deg2rad(45)
            else:
                phi_off_arr[ii] = 0
        else:
            raise NotImplementedError("The antenna mount type is not implemented!")

    field_rotation_angle_arr = np.zeros([len(uantennas), len(time_array), nfreq])

    for ii in range(len(uantennas)): # field rotation angle phi = f_el * theta_el + f_par * parang + phi_off
        for jj in range(len(time_array)):
            for kk in range(nfreq):
                field_rotation_angle_arr[ii, jj, kk] = f_el_arr[ii] * altitude_arr[ii][jj] + f_par_arr[ii] * parang_arr[ii][
                    jj] + phi_off_arr[ii]

    domain = ift.DomainTuple.make([ift.UnstructuredDomain(len(uantennas)), time_domain, ift.UnstructuredDomain(nfreq)])
    field_rotation_angle_field = ift.Field.from_raw(domain, field_rotation_angle_arr) #unit: [rad]

    return field_rotation_angle_field


def pol_cal_op_RR(config, obs, logamp_RCP_, phase_RCP_, Const_Dterm_logamp_RCP_, Const_Dterm_phase_RCP_, field_rotation_angle_field, parang_corrected=True, tmin=None, tmax=None):
    # gain RRRR, Dterm 11RR, Field 2j * 00ii, Field -2j * 0j0j, Dterm 1R1R*, gain RRRR*
    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    if tmin==None and tmax==None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    dom_pol_gain = ift.DomainTuple.make([ift.UnstructuredDomain(4), logamp_RCP_.target[0], logamp_RCP_.target[1]])
    Gain_Expander = ift.ContractionOperator(dom_pol_gain, 0).adjoint

    logamp_RRRR = Gain_Expander @ logamp_RCP_
    phase_RRRR = Gain_Expander @ phase_RCP_

    Expander_00RR = Expander_00DD(Const_Dterm_logamp_RCP_.target)
    Dterm_logamp_00RR = Expander_00RR @ Const_Dterm_logamp_RCP_
    Expander_0R0R = Expander_0D0D(Const_Dterm_logamp_RCP_.target)
    Dterm_logamp_0R0R = Expander_0R0R @ Const_Dterm_logamp_RCP_

    if parang_corrected == True:
        Field_rotation_i_scaling_2 = ift.Adder(2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_i_ = Field_rotation_i_scaling_2 @ Const_Dterm_phase_RCP_
        Dterm_phase_and_field_rotation_00RR = Expander_00RR @ Dterm_phase_and_field_rotation_i_

        Field_rotation_j_scaling_2 = ift.Adder(2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_j_ = Field_rotation_j_scaling_2 @ Const_Dterm_phase_RCP_
        Dterm_phase_and_field_rotation_0R0R = Expander_0R0R @ Dterm_phase_and_field_rotation_j_
    else:
        Dterm_phase_00RR = Expander_00RR @ Const_Dterm_phase_RCP_
        field_rotation_angle_array_i_NNPP = np.stack((-1*field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, field_rotation_angle_field.val))
        field_rotation_angle_field_i_NNPP = ift.Field.from_raw(domain=Dterm_phase_00RR.target, arr=field_rotation_angle_array_i_NNPP)
        Field_rotation_angle_op_i_NNPP = ift.Adder(field_rotation_angle_field_i_NNPP)
        Dterm_phase_and_field_rotation_00RR = Field_rotation_angle_op_i_NNPP @ Dterm_phase_00RR

        Dterm_phase_0R0R = Expander_0R0R @ Const_Dterm_phase_RCP_
        field_rotation_angle_array_j_PNPN = np.stack((field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, -1*field_rotation_angle_field.val))
        field_rotation_angle_field_j_PNPN = ift.Field.from_raw(domain=Dterm_phase_0R0R.target, arr=field_rotation_angle_array_j_PNPN)
        Field_rotation_angle_op_j_PNPN = ift.Adder(field_rotation_angle_field_j_PNPN)
        Dterm_phase_and_field_rotation_0R0R = Field_rotation_angle_op_j_PNPN @ Dterm_phase_0R0R


    pdom, _, fdom = obs.vis.domain

    reshaper = ift.DomainChangerAndReshaper(
        logamp_RRRR.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    logamp_i = reshaper @ logamp_RRRR
    phase_i = reshaper @ phase_RRRR

    logamp_j = reshaper @ logamp_RRRR
    phase_j = reshaper @ phase_RRRR

    reshaper2 = ift.DomainChangerAndReshaper(
        Dterm_logamp_00RR.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    Dterm_logamp_i = reshaper2 @ Dterm_logamp_00RR
    Dterm_phase_and_field_rotation_i = reshaper2 @ Dterm_phase_and_field_rotation_00RR

    Dterm_logamp_j = reshaper2 @ Dterm_logamp_0R0R
    Dterm_phase_and_field_rotation_j = reshaper2 @ Dterm_phase_and_field_rotation_0R0R

    cop1 = rve.CalibrationDistributor(logamp_i.target, obs.vis.domain, obs.ant1, obs.time, antenna_dct, None)
    cop2 = rve.CalibrationDistributor(logamp_j.target, obs.vis.domain, obs.ant2, obs.time, antenna_dct, None)

    res0 = cop1.real @ logamp_i + cop1.real @ Dterm_logamp_i + cop2.real @ Dterm_logamp_j + cop2.real @ logamp_j
    res1 = 1j * cop1.real @ phase_i + 1j * cop1.real @ Dterm_phase_and_field_rotation_i - 1j * cop2.real @ Dterm_phase_and_field_rotation_j - 1j * cop2.real @ phase_j

    return (res0 + res1).exp()


def pol_cal_op_RL(config, obs, logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_, Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_, field_rotation_angle_field, parang_corrected=True, tmin=None, tmax=None):
    # gain RRRR, Dterm 11RR, Field 2j * 00ii, Field 2j * j0j0, Dterm L1L1*, gain LLLL*
    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    if tmin==None and tmax==None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    dom_pol_gain = ift.DomainTuple.make([ift.UnstructuredDomain(4), logamp_RCP_.target[0], logamp_RCP_.target[1]])
    Gain_Expander = ift.ContractionOperator(dom_pol_gain, 0).adjoint

    logamp_RRRR = Gain_Expander @ logamp_RCP_
    phase_RRRR = Gain_Expander @ phase_RCP_

    logamp_LLLL = Gain_Expander @ logamp_LCP_
    phase_LLLL = Gain_Expander @ phase_LCP_

    Expander_00RR = Expander_00DD(Const_Dterm_logamp_RCP_.target)
    Dterm_logamp_00RR = Expander_00RR @ Const_Dterm_logamp_RCP_
    Expander_L0L0 = Expander_D0D0(Const_Dterm_logamp_LCP_.target)
    Dterm_logamp_L0L0 = Expander_L0L0 @ Const_Dterm_logamp_LCP_

    if parang_corrected == True:
        Field_rotation_i_scaling_2 = ift.Adder(2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_i_ = Field_rotation_i_scaling_2 @ Const_Dterm_phase_RCP_
        Dterm_phase_and_field_rotation_00RR = Expander_00RR @ Dterm_phase_and_field_rotation_i_

        Field_rotation_j_scaling_minus_2 = ift.Adder(-2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_j_ = Field_rotation_j_scaling_minus_2 @ Const_Dterm_phase_LCP_
        Dterm_phase_and_field_rotation_L0L0 = Expander_L0L0 @ Dterm_phase_and_field_rotation_j_
    else:
        Dterm_phase_00RR = Expander_00RR @ Const_Dterm_phase_RCP_
        field_rotation_angle_array_i_NNPP = np.stack((-1*field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, field_rotation_angle_field.val))
        field_rotation_angle_field_i_NNPP = ift.Field.from_raw(domain=Dterm_phase_00RR.target, arr=field_rotation_angle_array_i_NNPP)
        Field_rotation_angle_op_i_NNPP = ift.Adder(field_rotation_angle_field_i_NNPP)
        Dterm_phase_and_field_rotation_00RR = Field_rotation_angle_op_i_NNPP @ Dterm_phase_00RR

        Dterm_phase_L0L0 = Expander_L0L0 @ Const_Dterm_phase_LCP_
        field_rotation_angle_array_j_PNPN = np.stack((field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, -1*field_rotation_angle_field.val))
        field_rotation_angle_field_j_PNPN = ift.Field.from_raw(domain=Dterm_phase_L0L0.target, arr=field_rotation_angle_array_j_PNPN)
        Field_rotation_angle_op_j_PNPN = ift.Adder(field_rotation_angle_field_j_PNPN)
        Dterm_phase_and_field_rotation_L0L0 = Field_rotation_angle_op_j_PNPN @ Dterm_phase_L0L0

    pdom, _, fdom = obs.vis.domain

    reshaper = ift.DomainChangerAndReshaper(
        logamp_RRRR.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    logamp_i = reshaper @ logamp_RRRR
    phase_i = reshaper @ phase_RRRR

    logamp_j = reshaper @ logamp_LLLL
    phase_j = reshaper @ phase_LLLL

    reshaper2 = ift.DomainChangerAndReshaper(
        Dterm_logamp_00RR.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    Dterm_logamp_i = reshaper2 @ Dterm_logamp_00RR
    Dterm_phase_and_field_rotation_i = reshaper2 @ Dterm_phase_and_field_rotation_00RR

    Dterm_logamp_j = reshaper2 @ Dterm_logamp_L0L0
    Dterm_phase_and_field_rotation_j = reshaper2 @ Dterm_phase_and_field_rotation_L0L0


    cop1 = rve.CalibrationDistributor(logamp_i.target, obs.vis.domain, obs.ant1, obs.time, antenna_dct, None)
    cop2 = rve.CalibrationDistributor(logamp_j.target, obs.vis.domain, obs.ant2, obs.time, antenna_dct, None)

    res0 = cop1.real @ logamp_i + cop1.real @ Dterm_logamp_i + cop2.real @ Dterm_logamp_j + cop2.real @ logamp_j
    res1 = 1j * cop1.real @ phase_i + 1j * cop1.real @ Dterm_phase_and_field_rotation_i - 1j * cop2.real @ Dterm_phase_and_field_rotation_j - 1j * cop2.real @ phase_j

    return (res0 + res1).exp()


def pol_cal_op_LR(config, obs, logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_, Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_, field_rotation_angle_field, parang_corrected=True, tmin=None, tmax=None):
    # gain LLLL, Dterm LL11, Field -2j * ii00, Field -2j * 0j0j, Dterm 1R1R*, gain RRRR*
    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    if tmin == None and tmax == None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    dom_pol_gain = ift.DomainTuple.make([ift.UnstructuredDomain(4), logamp_RCP_.target[0], logamp_RCP_.target[1]])
    Gain_Expander = ift.ContractionOperator(dom_pol_gain, 0).adjoint

    logamp_LLLL = Gain_Expander @ logamp_LCP_
    phase_LLLL = Gain_Expander @ phase_LCP_

    logamp_RRRR = Gain_Expander @ logamp_RCP_
    phase_RRRR = Gain_Expander @ phase_RCP_

    Expander_LL00 = Expander_DD00(Const_Dterm_logamp_LCP_.target)
    Dterm_logamp_LL00 = Expander_LL00 @ Const_Dterm_logamp_LCP_
    Expander_0R0R = Expander_0D0D(Const_Dterm_logamp_RCP_.target)
    Dterm_logamp_0R0R = Expander_0R0R @ Const_Dterm_logamp_RCP_

    if parang_corrected == True:
        Field_rotation_i_scaling_minus_2 = ift.Adder(-2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_i_ = Field_rotation_i_scaling_minus_2 @ Const_Dterm_phase_LCP_
        Dterm_phase_and_field_rotation_LL00 = Expander_LL00 @ Dterm_phase_and_field_rotation_i_

        Field_rotation_j_scaling_2 = ift.Adder(2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_j_ = Field_rotation_j_scaling_2 @ Const_Dterm_phase_RCP_
        Dterm_phase_and_field_rotation_0R0R = Expander_0R0R @ Dterm_phase_and_field_rotation_j_
    else:
        Dterm_phase_LL00 = Expander_LL00 @ Const_Dterm_phase_LCP_
        field_rotation_angle_array_i_NNPP = np.stack((-1*field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, field_rotation_angle_field.val))
        field_rotation_angle_field_i_NNPP = ift.Field.from_raw(domain=Dterm_phase_LL00.target, arr=field_rotation_angle_array_i_NNPP)
        Field_rotation_angle_op_i_NNPP = ift.Adder(field_rotation_angle_field_i_NNPP)
        Dterm_phase_and_field_rotation_LL00 = Field_rotation_angle_op_i_NNPP @ Dterm_phase_LL00

        Dterm_phase_0R0R = Expander_0R0R @ Const_Dterm_phase_RCP_
        field_rotation_angle_array_j_PNPN = np.stack((field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, -1*field_rotation_angle_field.val))
        field_rotation_angle_field_j_PNPN = ift.Field.from_raw(domain=Dterm_phase_0R0R.target, arr=field_rotation_angle_array_j_PNPN)
        Field_rotation_angle_op_j_PNPN = ift.Adder(field_rotation_angle_field_j_PNPN)
        Dterm_phase_and_field_rotation_0R0R = Field_rotation_angle_op_j_PNPN @ Dterm_phase_0R0R

    pdom, _, fdom = obs.vis.domain

    reshaper = ift.DomainChangerAndReshaper(
        logamp_RRRR.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    logamp_i = reshaper @ logamp_RRRR
    phase_i = reshaper @ phase_RRRR

    logamp_j = reshaper @ logamp_LLLL
    phase_j = reshaper @ phase_LLLL

    reshaper2 = ift.DomainChangerAndReshaper(
        Dterm_logamp_LL00.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    Dterm_logamp_i = reshaper2 @ Dterm_logamp_LL00
    Dterm_phase_and_field_rotation_i = reshaper2 @ Dterm_phase_and_field_rotation_LL00

    Dterm_logamp_j = reshaper2 @ Dterm_logamp_0R0R
    Dterm_phase_and_field_rotation_j = reshaper2 @ Dterm_phase_and_field_rotation_0R0R


    cop1 = rve.CalibrationDistributor(logamp_i.target, obs.vis.domain, obs.ant1, obs.time, antenna_dct, None)
    cop2 = rve.CalibrationDistributor(logamp_j.target, obs.vis.domain, obs.ant2, obs.time, antenna_dct, None)

    res0 = cop1.real @ logamp_i + cop1.real @ Dterm_logamp_i + cop2.real @ Dterm_logamp_j + cop2.real @ logamp_j
    res1 = 1j * cop1.real @ phase_i + 1j * cop1.real @ Dterm_phase_and_field_rotation_i - 1j * cop2.real @ Dterm_phase_and_field_rotation_j - 1j * cop2.real @ phase_j

    return (res0 + res1).exp()


def pol_cal_op_LL(config, obs, logamp_LCP_, phase_LCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_LCP_, field_rotation_angle_field, parang_corrected=True, tmin=None, tmax=None):
    # gain LLLL, Dterm LL11, Field -2j * ii00, Field 2j * j0j0, Dterm L1L1*, gain LLLL*
    solint = config["gain_phase"].getint("solution_interval")
    zero_padding_factor = 2

    uantennas = rve.unique_antennas(obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    if tmin == None and tmax == None:
        tmin, tmax = rve.tmin_tmax(obs)
    assert tmin == 0.0
    time_domain = ift.RGSpace(
        ducc0.fft.good_size(int(zero_padding_factor * (tmax - tmin) / solint)), solint
    )

    dom_pol_gain = ift.DomainTuple.make([ift.UnstructuredDomain(4), logamp_LCP_.target[0], logamp_LCP_.target[1]])
    Gain_Expander = ift.ContractionOperator(dom_pol_gain, 0).adjoint

    logamp_LLLL = Gain_Expander @ logamp_LCP_
    phase_LLLL = Gain_Expander @ phase_LCP_

    Expander_LL00 = Expander_DD00(Const_Dterm_logamp_LCP_.target)
    Dterm_logamp_LL00 = Expander_LL00 @ Const_Dterm_logamp_LCP_
    Expander_L0L0 = Expander_D0D0(Const_Dterm_logamp_LCP_.target)
    Dterm_logamp_L0L0 = Expander_L0L0 @ Const_Dterm_logamp_LCP_

    if parang_corrected == True:
        Field_rotation_i_scaling_minus_2 = ift.Adder(-2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_i_ = Field_rotation_i_scaling_minus_2 @ Const_Dterm_phase_LCP_
        Dterm_phase_and_field_rotation_LL00 = Expander_LL00 @ Dterm_phase_and_field_rotation_i_

        Field_rotation_j_scaling_minus_2 = ift.Adder(-2*field_rotation_angle_field)
        Dterm_phase_and_field_rotation_j_ = Field_rotation_j_scaling_minus_2 @ Const_Dterm_phase_LCP_
        Dterm_phase_and_field_rotation_L0L0 = Expander_L0L0 @ Dterm_phase_and_field_rotation_j_
    else:
        Dterm_phase_LL00 = Expander_LL00 @ Const_Dterm_phase_LCP_
        field_rotation_angle_array_i_NNPP = np.stack((-1*field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, field_rotation_angle_field.val))
        field_rotation_angle_field_i_NNPP = ift.Field.from_raw(domain=Dterm_phase_LL00.target, arr=field_rotation_angle_array_i_NNPP)
        Field_rotation_angle_op_i_NNPP = ift.Adder(field_rotation_angle_field_i_NNPP)
        Dterm_phase_and_field_rotation_LL00 = Field_rotation_angle_op_i_NNPP @ Dterm_phase_LL00

        Dterm_phase_L0L0 = Expander_L0L0 @ Const_Dterm_phase_LCP_
        field_rotation_angle_array_j_PNPN = np.stack((field_rotation_angle_field.val, -1*field_rotation_angle_field.val,
                                                            field_rotation_angle_field.val, -1*field_rotation_angle_field.val))
        field_rotation_angle_field_j_PNPN = ift.Field.from_raw(domain=Dterm_phase_L0L0.target, arr=field_rotation_angle_array_j_PNPN)
        Field_rotation_angle_op_j_PNPN = ift.Adder(field_rotation_angle_field_j_PNPN)
        Dterm_phase_and_field_rotation_L0L0 = Field_rotation_angle_op_j_PNPN @ Dterm_phase_L0L0

    pdom, _, fdom = obs.vis.domain

    reshaper = ift.DomainChangerAndReshaper(
        logamp_LLLL.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    logamp_i = reshaper @ logamp_LLLL
    phase_i = reshaper @ phase_LLLL

    logamp_j = reshaper @ logamp_LLLL
    phase_j = reshaper @ phase_LLLL

    reshaper2 = ift.DomainChangerAndReshaper(
        Dterm_logamp_LL00.target,
        (
            pdom,
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            fdom
        )
    )

    Dterm_logamp_i = reshaper2 @ Dterm_logamp_LL00
    Dterm_phase_and_field_rotation_i = reshaper2 @ Dterm_phase_and_field_rotation_LL00

    Dterm_logamp_j = reshaper2 @ Dterm_logamp_L0L0
    Dterm_phase_and_field_rotation_j = reshaper2 @ Dterm_phase_and_field_rotation_L0L0


    cop1 = rve.CalibrationDistributor(logamp_i.target, obs.vis.domain, obs.ant1, obs.time, antenna_dct, None)
    cop2 = rve.CalibrationDistributor(logamp_j.target, obs.vis.domain, obs.ant2, obs.time, antenna_dct, None)

    res0 = cop1.real @ logamp_i + cop1.real @ Dterm_logamp_i + cop2.real @ Dterm_logamp_j + cop2.real @ logamp_j
    res1 = 1j * cop1.real @ phase_i + 1j * cop1.real @ Dterm_phase_and_field_rotation_i  - 1j * cop2.real @ Dterm_phase_and_field_rotation_j - 1j * cop2.real @ phase_j

    return (res0 + res1).exp()