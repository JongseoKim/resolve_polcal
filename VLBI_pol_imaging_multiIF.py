# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See theimport resolve as
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2023 Max-Planck-Society

import nifty8 as ift
import resolve as rve
import numpy as np
import matplotlib.pyplot as plt
import configparser
import sys
from resolve.data import ms_import as _ms_import


from src.sky_model import sky_model_diffuse
from src.cal_model import gain_ops, Dterm_ops, Const_Dterm_ops, Const_Dterm_ops_normal, get_field_rotation_angle_field, pol_cal_op_RR, pol_cal_op_RL, pol_cal_op_LR, pol_cal_op_LL
from src.likelihoods import likelihood_pol, PolarizationLikelihood, PolarizationLikelihood_combined
from src.utilities import get_BeginTime_UTC


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True


def _import_spw_on_full_row_grid(ms, spectral_window, polarizations):
    """Import one SPW without dropping fully flagged/zero-weight rows."""
    original_first_pass = _ms_import._first_pass

    def first_pass_keep_flagged_rows(ms, field, spw, channels, pol_indices,
                                     pol_summation, ignore_flags):
        # _first_pass normally removes both fully flagged rows and rows whose
        # weights are all zero. For multi-IF concatenation the row grid must
        # instead be determined solely by field and SPW. The second pass still
        # honors flags/weights, so unusable samples remain present with weight
        # zero and do not contribute to the likelihood.
        with _ms_import.ms_table(ms) as table:
            active_rows = np.logical_and(
                table.getcol("FIELD_ID") == field,
                table.getcol("DATA_DESC_ID") == spw,
            )
        active_channels = np.zeros(
            _ms_import._ms_nchannels(ms, spw), dtype=bool
        )
        active_channels[channels] = True
        return active_rows, active_channels

    _ms_import._first_pass = first_pass_keep_flagged_rows
    try:
        return rve.ms2observations(
            ms, "DATA", True, spectral_window, polarizations,
            ignore_flags=False,
        )[0]
    finally:
        _ms_import._first_pass = original_first_pass


def _assert_matching_row_grids(observations):
    reference = observations[0]._antpos
    for spw, observation in enumerate(observations[1:], start=1):
        candidate = observation._antpos
        matches = (
            len(candidate) == len(reference)
            and np.array_equal(candidate.ant1, reference.ant1)
            and np.array_equal(candidate.ant2, reference.ant2)
            and np.array_equal(candidate.time, reference.time)
            and np.array_equal(candidate.uvw, reference.uvw)
        )
        if not matches:
            raise RuntimeError(
                f"Spectral window {spw} ({len(candidate)} rows) does not have "
                f"the same antenna/time/UVW row grid as spectral window 0 "
                f"({len(reference)} rows); channel concatenation is unsafe."
            )


_, cfg_file = sys.argv
cfg = configparser.ConfigParser()
cfg.read(cfg_file)
cfg_base = cfg["base"]
nthreads_nifty = cfg_base.getint("nthreads_ift")


cfg_seed = cfg["seed"]
seed = cfg_seed.getint("random_seed")
ift.random.push_sseq_from_seed(seed)
ift.set_nthreads(nthreads_nifty)

cfg_observation = cfg["observation"]
data_path = cfg_observation["data_path"]
total_number_of_spectral_window = cfg_observation.getint("total_number_of_spectral_window")
spectral_window = cfg_observation.getint("spectral_window")
polarizations = cfg_observation["polarizations"]
parang_corrected = cfg_observation.getboolean("parang_corrected")
error_budget = cfg_observation.getfloat("error_budget")

#obs = rve.ms2observations(data_path, "DATA", True, spectral_window, polarizations)[0]
#obs = obs.to_double_precision()
#tmin, tmax = rve.tmin_tmax(obs)
#obs = obs.move_time(-tmin)

obs_list = []
vis_list = []
weight_list = []
freq_list = []
for ii in range(total_number_of_spectral_window):
    obs_list.append(_import_spw_on_full_row_grid(data_path, ii, polarizations))

_assert_matching_row_grids(obs_list)

for ii in range(total_number_of_spectral_window):
    vis_list.append(obs_list[ii].vis.val)
    weight_list.append(obs_list[ii].weight.val)
    freq_list.append(obs_list[ii].freq)

vis_new_ = np.concatenate(vis_list, axis=2)
weight_new_ = np.concatenate(weight_list, axis=2)
freq_new_ = np.concatenate(freq_list)
polarization_ = rve.Polarization((5,6,7,8)) # LL, RR : rve.Polarization((8,5)), pol imaging : rve.Polarization((5,6,7,8))
obs = rve.Observation(obs_list[0]._antpos, vis_new_, weight_new_, polarization_, freq_new_, auxiliary_tables=obs_list[0]._auxiliary_tables)
obs = obs.to_double_precision()
tmin, tmax = rve.tmin_tmax(obs)
obs = obs.move_time(-tmin)


uantennas = rve.unique_antennas(obs)
antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
station_table = obs._auxiliary_tables['ANTENNA']['STATION']


### adding systematic error budget
weight_old = obs.weight.val
new_weight = 1 / ((1 / np.sqrt(weight_old)) ** 2 + (
            error_budget * abs(obs.vis.val)) ** 2)  # 1/ (sigma**2 + (error_budget*|A|)**2)
obs._weight = new_weight




npix_x = cfg["sky"].getint("space npix x")
npix_y = cfg["sky"].getint("space npix y")
fov_x = cfg["sky"]["space fov x"]
fov_y = cfg["sky"]["space fov y"]

fov = np.array([rve.str2rad(fov_x), rve.str2rad(fov_y)])
npix = np.array([npix_x, npix_y])
sdom = ift.RGSpace(npix, fov / npix)
pdom = rve.PolarizationSpace(["I", "Q", "U", "V"])
position_space = rve.default_sky_domain(pdom=pdom, sdom=sdom)


sky_diffuse, logsky, additional_operators = sky_model_diffuse(cfg["sky"]) # logsky : a,b,c,d prior
sky = sky_diffuse

do_wgridding = cfg["base"].getboolean("do_wgridding")
epsilon = cfg["base"].getfloat("epsilon")
gaincal = cfg["gain_logamplitude"].getboolean("gaincal")
Const_Dterm = cfg["Dterm_logamplitude"]["Const_Dterm"]
reference_antenna_index = cfg["gain_phase"].get("reference_antenna_index", "None")

reference_antenna_index = reference_antenna_index.strip()
if reference_antenna_index == "None":
    reference_antenna_index = None
else:
    reference_antenna_index = cfg["gain_phase"].getint("reference_antenna_index")


logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_ = gain_ops(
    cfg,
    obs,
    gaincal=gaincal,
    zero_phase_rcp_antenna_index=reference_antenna_index,
    zero_phase_lcp_antenna_index=reference_antenna_index,
)


if Const_Dterm == "False":
    Dterm_logamp_, Dterm_phase_ = Dterm_ops(cfg, obs)
if Const_Dterm == "True":
    Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_ = Const_Dterm_ops(cfg, obs)


BeginTime_UTC_list, timestamp = get_BeginTime_UTC(cfg)
field_rotation_angle_field = get_field_rotation_angle_field(cfg, obs, BeginTime_UTC_list, timestamp)


'''
### Field rotation plots
Field_rotation_prior = Field_rotation_i_(ift.Field.from_raw(Field_rotation_i_.domain, np.zeros(Field_rotation_i_.domain.shape)))
solint = cfg["gain_phase"].getint("solution_interval")
time_arr = np.arange(0, solint*Field_rotation_i_.domain.shape[1], solint)
for ii in range(Field_rotation_i_.domain.shape[0]):
    plt.scatter(time_arr, np.rad2deg(Field_rotation_prior.val[ii]))
    plt.scatter(obs.time, np.zeros(obs.time.shape))
    plt.savefig(f"Field_rotation_angle_{ii}_{station_table[list(uantennas)[ii]]}.png")
    plt.close()
exit()
'''

pol_cal_op_RR = pol_cal_op_RR(cfg, obs, logamp_RCP_, phase_RCP_, Const_Dterm_logamp_RCP_, Const_Dterm_phase_RCP_, field_rotation_angle_field, parang_corrected=parang_corrected)
pol_cal_op_RL = pol_cal_op_RL(cfg, obs, logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_, Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_, field_rotation_angle_field, parang_corrected=parang_corrected)
pol_cal_op_LR = pol_cal_op_LR(cfg, obs, logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_, Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_, field_rotation_angle_field, parang_corrected=parang_corrected)
pol_cal_op_LL = pol_cal_op_LL(cfg, obs, logamp_LCP_, phase_LCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_LCP_, field_rotation_angle_field, parang_corrected=parang_corrected)


R = rve.InterferometryResponse(
    obs,
    sky.target,
    do_wgridding=False,
    epsilon=epsilon
)

Likelihood, Rs_pol = PolarizationLikelihood(obs, R, sky, pol_cal_op_RR, pol_cal_op_RL, pol_cal_op_LR, pol_cal_op_LL)



cfg_optimization = cfg["optimization"]
ic_sampling_iter_lim = cfg_optimization.getint("ic_samplilng_iter_lim")
ic_VL_BFGS_iter_lim_1 = cfg_optimization.getint("ic_VL_BFGS_iter_lim_1")
ic_VL_BFGS_iter_lim_2 = cfg_optimization.getint("ic_VL_BFGS_iter_lim_2")
ic_newton_iter_lim_3 = cfg_optimization.getint("ic_newton_iter_lim_3")
ic_newton_iter_lim_4 = cfg_optimization.getint("ic_newton_iter_lim_4")
ic_sampling_nl_iter_lim = cfg_optimization.getint("ic_sampling_nl_iter_lim")
n_iterations = cfg_optimization.getint("n_iterations")
n_samples_ = cfg_optimization.getint("n_samples")
output_directory = cfg_optimization["output_directory"]


ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit=ic_sampling_iter_lim)
ic_VL_BFGS_1 = ift.AbsDeltaEnergyController(name="VL_BFGS_1", deltaE=0.5,
                                         convergence_level=2, iteration_limit=ic_VL_BFGS_iter_lim_1)
ic_VL_BFGS_2 = ift.AbsDeltaEnergyController(name="VL_BFGS_2", deltaE=0.5,
                                         convergence_level=2, iteration_limit=ic_VL_BFGS_iter_lim_2)
ic_newton_3 = ift.AbsDeltaEnergyController(name="Newton_3", deltaE=0.5,
                                         convergence_level=2, iteration_limit=ic_newton_iter_lim_3)
ic_newton_4 = ift.AbsDeltaEnergyController(name="Newton_4", deltaE=0.5,
                                         convergence_level=2, iteration_limit=ic_newton_iter_lim_4)
ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                              deltaE=0.5, iteration_limit=ic_sampling_nl_iter_lim,
                                              convergence_level=2)
minimizer_1 = ift.VL_BFGS(ic_VL_BFGS_1)
minimizer_2 = ift.VL_BFGS(ic_VL_BFGS_2)
minimizer_3 = ift.NewtonCG(ic_newton_3)
minimizer_4 = ift.NewtonCG(ic_newton_4)
minimizer = lambda iiter : minimizer_1 if iiter < 5 else minimizer_2 if iiter < 10 else minimizer_3 if iiter < 15 else minimizer_4
minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

n_samples = lambda iiter: n_samples_ if iiter < 15 else 2 * n_samples_
initial_position = ift.from_random(Likelihood.domain, std=0.3)

def inspect_callback(sl, iglobal):

    sky_posterior_mean = sl.average(sky)
    StokesI_sky = sky_posterior_mean.val.T[:, :, 0, 0, 0]
    StokesQ_sky = sky_posterior_mean.val.T[:, :, 0, 0, 1]
    StokesU_sky = sky_posterior_mean.val.T[:, :, 0, 0, 2]
    StokesV_sky = sky_posterior_mean.val.T[:, :, 0, 0, 3]
    Linear_pol = abs(StokesQ_sky + 1j * StokesU_sky)
    EVPA = np.rad2deg(0.5 * np.arctan2(StokesU_sky, StokesQ_sky))

    Dterm_posterior_mean_RCP = sl.average(np.exp(Const_Dterm_logamp_RCP_ + 1j * Const_Dterm_phase_RCP_))
    Dterm_posterior_mean_LCP = sl.average(np.exp(Const_Dterm_logamp_LCP_ + 1j * Const_Dterm_phase_LCP_))
    Dterm_posterior_mean_RCP = Dterm_posterior_mean_RCP.val[:,0]
    Dterm_posterior_mean_LCP = Dterm_posterior_mean_LCP.val[:,0]

    filename = f"{output_directory}/{output_directory}_Dterm_posterior_mean_{iglobal}.png"
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(6, 12))

    for ii in np.arange(int(len(uantennas))):
        ax1.scatter(Dterm_posterior_mean_RCP.real[ii], Dterm_posterior_mean_RCP.imag[ii],
                    label=f'{station_table[list(uantennas)[ii]]}')

    ax1.set_title('RCP Dterms', fontsize=15)
    ax1.set_xlabel('real')
    ax1.set_ylabel('imag')
    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-0.1, 0.1)
    ax1.legend(loc='lower left')

    for ii in np.arange(int(len(uantennas))):
        ax2.scatter(Dterm_posterior_mean_LCP.real[ii], Dterm_posterior_mean_LCP.imag[ii],
                    label=f'{station_table[list(uantennas)[ii]]}')

    ax2.set_title('LCP Dterms', fontsize=15)
    ax2.set_xlabel('real')
    ax2.set_ylabel('imag')
    ax2.set_xlim(-0.1, 0.1)
    ax2.set_ylim(-0.1, 0.1)
    ax2.legend(loc='lower left')
    plt.savefig(filename)
    print(f"Saved results as {filename}.")
    plt.close()

    plt.imsave(f"{output_directory}/StokesI_sky_{iglobal}_logscale.png", np.log(StokesI_sky), origin="lower", cmap="inferno")
    if iglobal > 10:
        plt.imsave(f"{output_directory}/StokesQ_sky_{iglobal}.png", StokesQ_sky, origin="lower", cmap="bwr")
        plt.imsave(f"{output_directory}/StokesU_sky_{iglobal}.png", StokesU_sky, origin="lower", cmap="bwr")
        plt.imsave(f"{output_directory}/StokesV_sky_{iglobal}.png", StokesV_sky, origin="lower", cmap="bwr")
        plt.imsave(f"{output_directory}/Linear_pol_{iglobal}_logscale.png", np.log(Linear_pol), origin="lower", cmap="gist_rainbow")





samples = ift.optimize_kl(
    Likelihood,
    n_iterations,
    n_samples,
    minimizer,
    ic_sampling,
    None,
    export_operator_outputs={"sky":sky, "logamp_RCP":logamp_RCP_, "phase_RCP":phase_RCP_, "logamp_LCP":logamp_LCP_, "phase_LCP":phase_LCP_,
                             "Dterm_logamp_RCP":Const_Dterm_logamp_RCP_, "Dterm_phase_RCP":Const_Dterm_phase_RCP_,
                             "Dterm_logamp_LCP":Const_Dterm_logamp_LCP_, "Dterm_phase_LCP":Const_Dterm_phase_LCP_},
    output_directory=output_directory,
    inspect_callback=inspect_callback,
    #initial_position=initial_position,
    comm=comm,
    resume=True
)
