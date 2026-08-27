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
import scipy.stats
from matplotlib import colors, cm
import configparser
import sys
import h5py
from resolve.data import ms_import as _ms_import

from src.sky_model import sky_model_diffuse
from src.cal_model import gain_ops, Dterm_ops, Const_Dterm_ops, get_field_rotation_angle_field, pol_cal_op_RR, pol_cal_op_RL, pol_cal_op_LR, pol_cal_op_LL
from src.likelihoods import likelihood_pol, PolarizationLikelihood, PolarizationLikelihood_combined
from src.utilities import get_BeginTime_UTC


def _import_spw_on_full_row_grid(ms, spectral_window, polarizations):
    """Import one SPW without dropping fully flagged/zero-weight rows."""
    original_first_pass = _ms_import._first_pass

    def first_pass_keep_flagged_rows(ms, field, spw, channels, pol_indices,
                                     pol_summation, ignore_flags):
        # Multi-IF concatenation requires every SPW to retain the same row
        # grid. Flags are still honored below, so unusable samples get zero
        # weight and do not contribute to subsequent calculations.
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
cfg = configparser.ConfigParser(allow_no_value=True)
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
antenna_mount_type = obs._auxiliary_tables["ANTENNA"]["MOUNT"]


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
'''
do_wgridding = cfg["base"].getboolean("do_wgridding")
epsilon = cfg["base"].getfloat("epsilon")
gaincal = cfg["gain_logamplitude"].getboolean("gaincal")
Const_Dterm = cfg["Dterm_logamplitude"]["Const_Dterm"]
zero_phase_lcp_antenna_index = cfg["gain_phase"]["zero_phase_lcp_antenna_index"]

if zero_phase_lcp_antenna_index == "None":
    zero_phase_lcp_antenna_index = None
else:
    zero_phase_lcp_antenna_index = cfg["gain_phase"].getint("zero_phase_lcp_antenna_index")


logamp_RCP_, logamp_LCP_, phase_RCP_, phase_LCP_ = gain_ops(
    cfg,
    obs,
    gaincal=gaincal,
    zero_phase_lcp_antenna_index=zero_phase_lcp_antenna_index,
)


if Const_Dterm == "False":
    Dterm_logamp_, Dterm_phase_ = Dterm_ops(cfg, obs)
if Const_Dterm == "True":
    Const_Dterm_logamp_RCP_, Const_Dterm_logamp_LCP_, Const_Dterm_phase_RCP_, Const_Dterm_phase_LCP_ = Const_Dterm_ops(cfg, obs)


BeginTime_UTC_list, timestamp = get_BeginTime_UTC(cfg)
field_rotation_angle_field = get_field_rotation_angle_field(cfg, obs, BeginTime_UTC_list, timestamp)


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
'''

cfg_optimization = cfg["optimization"]
n_iterations = cfg_optimization.getint("n_iterations")
n_samples_ = cfg_optimization.getint("n_samples")
output_directory = cfg_optimization["output_directory"]
solution_interval = cfg["gain_phase"].getint("solution_interval")


dx = rve.str2rad(fov_x) / npix_x
dy = rve.str2rad(fov_y) / npix_y
Nx = npix_x
Ny = npix_y
Tx = dx / rve.AS2RAD * 1e3 # from rad to mas
Ty = dy / rve.AS2RAD * 1e3 # from rad to mas
x = np.linspace(0, Nx * Tx, Nx, endpoint=False)
y = np.linspace(0, Ny * Ty, Ny, endpoint=False)
[X, Y] = np.meshgrid(x, y)
X_2 = X[::2, ::2]
Y_2 = Y[::2, ::2]
X_5 = X[::5, ::5]
Y_5 = Y[::5, ::5]
X_10 = X[::10, ::10]
Y_10 = Y[::10, ::10]


'''
### GPCAL sim Dterms from csv
import pandas as pd
data = pd.read_csv("Data/true_dterms.csv", sep="\t")

station_table_ordered = ['BR', 'FD', 'HN', 'KP', 'LA', 'MK', 'NL', 'OV', 'PT', 'SC']

filename = f"POLSIM_synthetic_data_GPCAL_Dterms.png"
fig, ((ax1) ,(ax2)) = plt.subplots(2, 1, figsize=(6, 12))

print(data.DR)

for ii in np.arange(int(len(uantennas))):
    DR = np.complex128(data.DR[ii])
    print(DR)
    ax1.scatter(DR.real, DR.imag,
             label=f'{station_table_ordered[list(uantennas)[ii]]}')

ax1.set_title('RCP Dterms', fontsize=15)
ax1.set_xlabel('real')
ax1.set_ylabel('imag')
ax1.set_xlim(-0.2, 0.2)
ax1.set_ylim(-0.2, 0.2)
ax1.legend(loc='lower left')

print(data.DL)

for ii in np.arange(int(len(uantennas))):
    DL = np.complex128(data.DL[ii])
    print(DL)
    ax2.scatter(DL.real, DL.imag,
             label=f'{station_table_ordered[list(uantennas)[ii]]}')

ax2.set_title('LCP Dterms', fontsize=15)
ax2.set_xlabel('real')
ax2.set_ylabel('imag')
ax2.set_xlim(-0.2, 0.2)
ax2.set_ylim(-0.2, 0.2)
ax2.legend(loc='lower left')
plt.savefig(filename)
print(f"Saved results as {filename}.")
plt.close()
exit()
'''


'''
### GPCAL Fig8 Dterms from csv
import pandas as pd
data = pd.read_csv("Data/Figure8_dterms.csv", sep="\t")

#for ii in range(10):
#    Dterm_RCP = np.complex128(data[data["antennas"] == station_table[ii]].DRArr.to_list())
#    print(Dterm_RCP[0])

filename = f"Fig8_3C273_GPCAL_Dterms.png"
fig, ((ax1) ,(ax2)) = plt.subplots(2, 1, figsize=(6, 12))

for ii in np.arange(int(len(uantennas))):
    Dterm_RCP = np.mean(np.complex128(data[data["antennas"] == station_table[ii]].DRArr.to_list()))
    print(Dterm_RCP)
    ax1.scatter(Dterm_RCP.real, Dterm_RCP.imag,
             label=f'{station_table[list(uantennas)[ii]]}')

ax1.set_title('RCP Dterms', fontsize=15)
ax1.set_xlabel('real')
ax1.set_ylabel('imag')
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.legend(loc='lower left')

for ii in np.arange(int(len(uantennas))):
    Dterm_LCP = np.mean(np.complex128(data[data["antennas"] == station_table[ii]].DLArr.to_list()))
    ax2.scatter(Dterm_LCP.real, Dterm_LCP.imag,
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

for ii in range(10):
    Dterm_RCP = np.complex128(data[data["antennas"] == station_table[ii]].DRArr.to_list())
    print(Dterm_RCP[0])

filename = f"Fig8_3C273_GPCAL_Dterms_IF1.png"
fig, ((ax1) ,(ax2)) = plt.subplots(2, 1, figsize=(6, 12))

for ii in np.arange(int(len(uantennas))):
    Dterm_RCP = np.complex128(data[data["antennas"] == station_table[ii]].DRArr.to_list())
    print(Dterm_RCP[0])
    ax1.scatter(Dterm_RCP[0].real, Dterm_RCP[0].imag,
             label=f'{station_table[list(uantennas)[ii]]}')

ax1.set_title('RCP Dterms', fontsize=15)
ax1.set_xlabel('real')
ax1.set_ylabel('imag')
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.legend(loc='lower left')

for ii in np.arange(int(len(uantennas))):
    Dterm_LCP = np.complex128(data[data["antennas"] == station_table[ii]].DLArr.to_list())
    ax2.scatter(Dterm_LCP[0].real, Dterm_LCP[0].imag,
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
exit()
'''


### Sky plotter
with h5py.File(f"{output_directory}/sky/last.hdf5", 'r') as hdf:
    if n_samples_ == 0:

        arr_sample0 = hdf['samples']['0']
        sky_MAP = ift.makeField(position_space, arr_sample0)
        sky_MAP_Jy_mas2 = sky_MAP / (180 / np.pi * 3600 * 1000) ** 2
        # ift.single_plot(sky_MAP)
        StokesI_sky = sky_MAP.val.T[:, :, 0, 0, 0]
        StokesQ_sky = sky_MAP.val.T[:, :, 0, 0, 1]
        StokesU_sky = sky_MAP.val.T[:, :, 0, 0, 2]
        StokesV_sky = sky_MAP.val.T[:, :, 0, 0, 3]
        StokesI_sky_Jy_mas2 = sky_MAP_Jy_mas2.val.T[:, :, 0, 0, 0]
        StokesQ_sky_Jy_mas2 = sky_MAP_Jy_mas2.val.T[:, :, 0, 0, 1]
        StokesU_sky_Jy_mas2 = sky_MAP_Jy_mas2.val.T[:, :, 0, 0, 2]
        StokesV_sky_Jy_mas2 = sky_MAP_Jy_mas2.val.T[:, :, 0, 0, 3]

        Linear_pol = abs(StokesQ_sky + 1j * StokesU_sky)
        Frac_linear_pol = Linear_pol / StokesI_sky
        EVPA = np.rad2deg(0.5 * np.arctan2(StokesU_sky, StokesQ_sky))

        # absolute EVPA correction for 2017_01_28 3C273
        if data_path == "Data/3C273_folder/1226+023.u.2017_01_28_uvaver10_raw.ms" or "Data/MOJAVE_BL229AE_20170128/1226+023.u.2017_01_28_raw_edt_freqavg.ms":
            EVPA = EVPA - 35.5

        EVPA_rad = np.deg2rad(EVPA)
        EVPA_mask = EVPA_rad


        # calculate noise level
        x = []
        for ii in np.arange(int(npix_y/4)):
            for jj in np.arange(int(npix_y/4)):
                x.append(StokesI_sky[ii,jj])
        sigma = np.std(x, ddof=1)


        for ii in np.arange(npix_y):
            for jj in np.arange(npix_x):
                if Linear_pol[ii, jj] < 0.002*np.max(StokesI_sky): #0.2 percent of StokesI
                    EVPA_mask[ii, jj] = np.nan

        u = -1 * np.sin(EVPA_mask)
        v = np.cos(EVPA_mask)
        u_2 = u[::2, ::2]
        v_2 = v[::2, ::2]
        u_5 = u[::5, ::5]
        v_5 = v[::5, ::5]
        u_10 = u[::10, ::10]
        v_10 = v[::10, ::10]

        ### Plotting EVPA
        filename = f"{output_directory}/{output_directory}_MAP_EVPA.png"
        #plt.figure(figsize=(16,6))
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8), dpi=300)

        q1 = ax1.quiver(X_5, Y_5, u_5, v_5, scale=20, scale_units='inches', width=0.0005, headwidth=0, headlength=0, headaxislength=0)
        I_contour_ax4 = ax1.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(15*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax1.set_aspect('equal')
        ax1.set_title('EVPA', fontsize=30)
        ax1.tick_params(labelsize=20)
        ax1.set_xlabel('[mas]', fontsize=20)
        ax1.set_ylabel('[mas]', fontsize=20)
        #cbar_ax4 = fig.colorbar(SkyV, ax=ax4)
        #cbar_ax4.remove()
        plt.savefig(filename)
        plt.close()
        #exit()

        ### Fitsfile
        StokesI_sky_Jy_mas2_field = ift.makeField(sdom, StokesI_sky_Jy_mas2.T)
        StokesQ_sky_Jy_mas2_field = ift.makeField(sdom, StokesQ_sky_Jy_mas2.T)
        StokesU_sky_Jy_mas2_field = ift.makeField(sdom, StokesU_sky_Jy_mas2.T)
        rve.field2fits(StokesI_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesI_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)
        rve.field2fits(StokesQ_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesQ_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)
        rve.field2fits(StokesU_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesU_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)


        ### Plotting
        filename = f"{output_directory}/{output_directory}_MAP_sky.png"
        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(30,10))

        skyI = ax1.pcolormesh(X, Y, StokesI_sky, norm=colors.LogNorm(), cmap=cm.inferno)
        ax1.set_aspect('equal')
        ax1.set_title('Stokes I', fontsize=20)
        cbar_ax1 = fig.colorbar(skyI, ax=ax1)
        cbar_ax1.set_label('Jy/str')

        frac_lin_pol = ax2.pcolormesh(X, Y, Frac_linear_pol, cmap=cm.gist_rainbow)
        I_contour_ax2 = ax2.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax2.set_aspect('equal')
        ax2.set_title('Fractional linear polarization', fontsize=20)
        cbar_ax2 = fig.colorbar(frac_lin_pol, ax=ax2)

        lin_pol = ax3.pcolormesh(X, Y, Linear_pol, norm=colors.LogNorm(), cmap=cm.gist_rainbow)
        I_contour_ax3 = ax3.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax3.set_aspect('equal')
        ax3.set_title('Linear polarization', fontsize=20)
        cbar_ax3 = fig.colorbar(lin_pol, ax=ax3)
        cbar_ax3.set_label('Jy/str')

        q1 = ax4.quiver(X_5, Y_5, u_5, v_5, scale=14, scale_units='inches', width=0.001, headwidth=0, headlength=0, headaxislength=0)
        I_contour_ax4 = ax4.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax4.set_aspect('equal')
        ax4.set_title('EVPA', fontsize=20)
        #cbar_ax6 = fig.colorbar(sky_EVPA, ax=ax6)
        #cbar_ax6.remove()

        skyV = ax5.pcolormesh(X, Y, StokesV_sky, cmap=cm.bwr)
        ax5.set_aspect('equal')
        ax5.set_title('Stokes V', fontsize=20)
        cbar_ax5 = fig.colorbar(skyV, ax=ax5)
        cbar_ax5.set_label('Jy/str')

        sky_EVPA = ax6.pcolormesh(X, Y, EVPA, cmap=cm.hsv)
        #I_contour_ax5 = ax5.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(15 * sigma), 2, num=3),linewidths=0.3)
        ax6.set_aspect('equal')
        ax6.set_title('EVPA', fontsize=20)
        cbar_ax6 = fig.colorbar(sky_EVPA, ax=ax6)
        cbar_ax6.set_label('deg')

        plt.savefig(filename)


    else:
        sky_samplelist = []
        sky_arraylist = []
        for ii in range(len(hdf["samples"])):
            arr_sample = hdf['samples'][f'{ii}']
            sky_array = np.array(arr_sample)

            sky_arraylist.append(sky_array)
            sky_sample = ift.makeField(position_space, arr_sample)
            sky_samplelist.append(sky_sample)


        sky_pos_mean = np.mean(sky_arraylist, axis=0)
        sky_pos_mean_Jy_mas2 = sky_pos_mean / (180 / np.pi * 3600 * 1000) ** 2
        StokesI_sky = sky_pos_mean.T[:, :, 0, 0, 0]
        StokesQ_sky = sky_pos_mean.T[:, :, 0, 0, 1]
        StokesU_sky = sky_pos_mean.T[:, :, 0, 0, 2]
        StokesV_sky = sky_pos_mean.T[:, :, 0, 0, 3]
        StokesI_sky_Jy_mas2 = sky_pos_mean_Jy_mas2.T[:, :, 0, 0, 0]
        StokesQ_sky_Jy_mas2 = sky_pos_mean_Jy_mas2.T[:, :, 0, 0, 1]
        StokesU_sky_Jy_mas2 = sky_pos_mean_Jy_mas2.T[:, :, 0, 0, 2]
        StokesV_sky_Jy_mas2 = sky_pos_mean_Jy_mas2.T[:, :, 0, 0, 3]

        linear_pol_posterior = []
        frac_linear_pol_posterior = []
        frac_circular_pol_posterior = []
        EVPA_posterior = []

        # calculate noise level
        x = []
        for ii in np.arange(int(npix_y/4)):
            for jj in np.arange(int(npix_y/4)):
                x.append(StokesI_sky[ii,jj])
        sigma = np.std(x, ddof=1)
        sigma_Jy_mas2 = sigma / (180 / np.pi * 3600 * 1000) ** 2

        for index, sky_samples in enumerate(sky_samplelist):
            linear_pol_sample = abs(sky_samples.val.T[:, :, 0, 0, 1] + 1j * sky_samples.val.T[:, :, 0, 0, 2]) # P = |Q +iU|
            circular_pol_sample = sky_samples.val.T[:, :, 0, 0, 3] # Stokes V

            linear_pol_posterior.append(linear_pol_sample)
            frac_linear_pol_sample = linear_pol_sample / sky_samples.val.T[:, :, 0, 0, 0]
            frac_linear_pol_posterior.append(frac_linear_pol_sample)
            frac_circular_pol_sample = circular_pol_sample / sky_samples.val.T[:, :, 0, 0, 0]
            frac_circular_pol_posterior.append(frac_circular_pol_sample)

            EVPA_sample = np.rad2deg(0.5 * np.arctan2(sky_samples.val.T[:, :, 0, 0, 2], sky_samples.val.T[:, :, 0, 0, 1]))
            EVPA_posterior.append(EVPA_sample)



        Linear_pol_postmean = np.mean(linear_pol_posterior, axis=0)
        Linear_pol_postmean_Jy_mas2 = Linear_pol_postmean / (180 / np.pi * 3600 * 1000) ** 2

        Frac_linear_pol_postmean = np.mean(frac_linear_pol_posterior, axis=0)
        Frac_circular_pol_postmean = np.mean(frac_circular_pol_posterior, axis=0)
        EVPA_postmean = np.mean(EVPA_posterior, axis=0)
        EVPA_postmean_circmean = scipy.stats.circmean(EVPA_posterior, high=90.0, low=-90.0, axis=0)
        EVPA_poststd_circstd = scipy.stats.circstd(EVPA_posterior, high=90.0, low=-90.0, axis=0)

        #TODO check EVPA_postmean
        '''
        print(EVPA_posterior)
        for ii in range(20):
            print(EVPA_posterior[ii][0,0])

        #plt.imshow(EVPA_postmean)
        plt.imshow(EVPA_postmean_circmean)
        plt.show()
        exit()
        '''

        # absolute EVPA correction for 2017_01_28 3C273
        if data_path == "Data/3C273_folder/1226+023.u.2017_01_28_raw_edt.ms" or "Data/MOJAVE_BL229AE_20170128/1226+023.u.2017_01_28_raw_edt_freqavg.ms":
            EVPA_postmean = EVPA_postmean - 35.5

        EVPA_std = np.std(EVPA_posterior, axis=0)
        EVPA_std_nomask = np.std(EVPA_posterior, axis=0)
        EVPA_poststd_nomask_circstd = scipy.stats.circstd(EVPA_posterior, high=90.0, low=-90.0, axis=0)
        EVPA_rad = np.deg2rad(EVPA_postmean)
        EVPA_mask = EVPA_rad
        EVPA_rad_circmean = np.deg2rad(EVPA_postmean_circmean)
        EVPA_mask_circmean = EVPA_rad_circmean



        for ii in np.arange(npix_y):
            for jj in np.arange(npix_x):
                if StokesI_sky[ii, jj] < 0.005 * np.max(StokesI_sky): # 0.5% of I_peak
                    EVPA_mask[ii, jj] = np.nan
                    EVPA_mask_circmean[ii, jj] = np.nan

        EVPA_mask_std = EVPA_std
        EVPA_mask_std_circmean = EVPA_poststd_circstd

        for ii in np.arange(npix_y):
            for jj in np.arange(npix_x):
                if StokesI_sky[ii, jj] < 0.005 * np.max(StokesI_sky):
                    EVPA_mask_std[ii, jj] = np.nan
                    EVPA_mask_std_circmean[ii, jj] = np.nan

        #u = -1 * np.sin(EVPA_mask)
        #v = np.cos(EVPA_mask)
        u = -1 * np.sin(EVPA_mask_circmean)
        v = np.cos(EVPA_mask_circmean)
        u_2 = u[::2, ::2]
        v_2 = v[::2, ::2]
        u_5 = u[::5, ::5]
        v_5 = v[::5, ::5]
        u_10 = u[::10, ::10]
        v_10 = v[::10, ::10]

        ### Plotting EVPA
        filename = f"{output_directory}/{output_directory}_MGVI_EVPA_pos_mean_circmean.png"
        #plt.figure(figsize=(16,6))
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

        q1 = ax1.quiver(X_2, Y_2, u_2, v_2, scale=20, scale_units='inches', width=0.001, headwidth=0, headlength=0, headaxislength=0)
        I_contour_ax4 = ax1.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax1.set_aspect('equal')
        ax1.set_title('EVPA', fontsize=30)
        ax1.tick_params(labelsize=20)
        ax1.set_xlabel('[mas]', fontsize=20)
        ax1.set_ylabel('[mas]', fontsize=20)
        #cbar_ax4 = fig.colorbar(SkyV, ax=ax4)
        #cbar_ax4.remove()
        plt.savefig(filename)
        plt.close()
        #exit()

        ### Fitsfile
        StokesI_sky_Jy_mas2_field = ift.makeField(sdom, StokesI_sky_Jy_mas2.T)
        StokesQ_sky_Jy_mas2_field = ift.makeField(sdom, StokesQ_sky_Jy_mas2.T)
        StokesU_sky_Jy_mas2_field = ift.makeField(sdom, StokesU_sky_Jy_mas2.T)
        StokesV_sky_Jy_mas2_field = ift.makeField(sdom, StokesV_sky_Jy_mas2.T)
        EVPA_postmean_field = ift.makeField(sdom, EVPA_postmean.T)
        EVPA_std_field = ift.makeField(sdom, EVPA_std_nomask.T)
        EVPA_postmean_field_circmean = ift.makeField(sdom, EVPA_postmean_circmean.T)
        EVPA_std_field_circstd = ift.makeField(sdom, EVPA_poststd_nomask_circstd.T)

        rve.field2fits(StokesI_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesI_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)
        rve.field2fits(StokesQ_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesQ_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)
        rve.field2fits(StokesU_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesU_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)
        rve.field2fits(StokesV_sky_Jy_mas2_field, f"{output_directory}/{output_directory}_StokesV_MGVI_{npix_x}_{npix_y}_Jy_mas2.fits", True, direction=obs.direction)
        rve.field2fits(EVPA_postmean_field_circmean, f"{output_directory}/{output_directory}_EVPA_postmean_MGVI_{npix_x}_{npix_y}_circmean.fits", True, direction=obs.direction)
        rve.field2fits(EVPA_std_field_circstd, f"{output_directory}/{output_directory}_EVPA_std_MGVI_{npix_x}_{npix_y}_circstd.fits", True, direction=obs.direction)

        #rve.field2fits(Linear_pol_postmean_Jy_mas2, f"M87_GMVA_ALMA_86GHz_Linear_pol_MGVI_{npix_x}_{npix_y}_Jy_mas2", True, direction=obs.direction)
        #rve.field2fits(StokesI_sky_Jy_mas2, f"M87_GMVA_ALMA_86GHz_StokesI_MGVI_{npix_x}_{npix_y}_Jy_mas2", True, direction=obs.direction)


        ### Plotting
        filename = f"{output_directory}/{output_directory}_MGVI_sky_circmean.png"
        fig, ((ax1, ax2, ax3), (ax5, ax4, ax6)) = plt.subplots(2, 3, figsize=(30,20))

        skyI = ax1.pcolormesh(X, Y, StokesI_sky_Jy_mas2, norm=colors.LogNorm(vmin=5*sigma_Jy_mas2), cmap=cm.inferno)
        ax1.set_aspect('equal')
        ax1.set_title('Stokes I', fontsize=30)
        ax1.tick_params(labelsize=20)
        ax1.set_xlabel('[mas]', fontsize=20)
        ax1.set_ylabel('[mas]', fontsize=20)
        cbar_ax1 = fig.colorbar(skyI, ax=ax1)
        cbar_ax1.ax.tick_params(labelsize=20)
        cbar_ax1.set_label('[Jy/mas2]', fontsize=20)


        frac_lin_pol = ax2.pcolormesh(X, Y, Frac_linear_pol_postmean, cmap=cm.gist_rainbow)
        I_contour_ax2 = ax2.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax2.set_aspect('equal')
        ax2.set_title('Fractional linear polarization', fontsize=30)
        ax2.tick_params(labelsize=20)
        ax2.set_xlabel('[mas]', fontsize=20)
        ax2.set_ylabel('[mas]', fontsize=20)
        cbar_ax2 = fig.colorbar(frac_lin_pol, ax=ax2)
        cbar_ax2.ax.tick_params(labelsize=20)


        SkyV = ax3.pcolormesh(X, Y, StokesV_sky_Jy_mas2, cmap=cm.bwr)
        I_contour_ax3 = ax3.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax3.set_aspect('equal') #TODO fix the oversized plot
        ax3.set_title('Linear polarization', fontsize=30)
        #ax3.tick_params(labelsize=20)
        #ax3.set_xlabel('[mas]', fontsize=20)
        #ax3.set_ylabel('[mas]', fontsize=20)
        cbar_ax3 = fig.colorbar(SkyV, ax=ax3)
        #cbar_ax3.ax.tick_params(labelsize=20)
        #cbar_ax3.set_label('[Jy/mas2]', fontsize=20)


        q1 = ax4.quiver(X_5, Y_5, u_5, v_5, scale=7, scale_units='inches', width=0.001, headwidth=0, headlength=0, headaxislength=0)
        I_contour_ax4 = ax4.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax4.set_aspect('equal')
        ax4.set_title('EVPA', fontsize=30)
        ax4.tick_params(labelsize=20)
        ax4.set_xlabel('[mas]', fontsize=20)
        ax4.set_ylabel('[mas]', fontsize=20)
        cbar_ax4 = fig.colorbar(SkyV, ax=ax4)
        cbar_ax4.remove()

        lin_pol = ax5.pcolormesh(X, Y, Linear_pol_postmean_Jy_mas2, norm=colors.LogNorm(), cmap=cm.gist_rainbow)
        I_contour_ax5 = ax5.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(10*sigma), np.log10(np.max(StokesI_sky)), num=3), linewidths=0.3)
        ax5.set_aspect('equal')
        ax5.set_title('Linear polarization', fontsize=30)
        ax5.tick_params(labelsize=20)
        ax5.set_xlabel('[mas]', fontsize=20)
        ax5.set_ylabel('[mas]', fontsize=20)
        cbar_ax5 = fig.colorbar(lin_pol, ax=ax5)
        cbar_ax5.ax.tick_params(labelsize=20)
        cbar_ax5.set_label('[Jy/mas2]', fontsize=20)

        EVPA_error = ax6.pcolormesh(X, Y, EVPA_mask_std_circmean, cmap=cm.viridis)
        ax6.set_aspect('equal')
        ax6.set_title('EVPA std', fontsize=30)
        ax6.tick_params(labelsize=20)
        ax6.set_xlabel('[mas]', fontsize=20)
        ax6.set_ylabel('[mas]', fontsize=20)
        cbar_ax6 = fig.colorbar(EVPA_error, ax=ax6)
        cbar_ax6.ax.tick_params(labelsize=20)
        cbar_ax6.set_label('[deg]', fontsize=20)


        #sky_EVPA = ax6.pcolormesh(X, Y, EVPA_postmean, cmap=cm.hsv)
        #I_contour_ax5 = ax5.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(15 * sigma), 2, num=3),linewidths=0.3)
        #ax6.set_aspect('equal')
        #ax6.set_title('EVPA', fontsize=20)
        #cbar_ax6 = fig.colorbar(sky_EVPA, ax=ax6)
        #cbar_ax6.set_label('deg')
        
        plt.savefig(filename)


        ### Compact region sky plotter

        # 0.25mas x 0.25mas
        X = X[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        Y = Y[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        StokesI_sky = StokesI_sky[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        StokesQ_sky = StokesQ_sky[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        StokesU_sky = StokesU_sky[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        StokesV_sky = StokesV_sky[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        Linear_pol_postmean = Linear_pol_postmean[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        Frac_linear_pol_postmean = Frac_linear_pol_postmean[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        EVPA_postmean = EVPA_postmean_circmean[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]
        EVPA_mask_circmean = EVPA_mask_circmean[int(7 / 16 * npix_y):int(9 / 16 * npix_y), int(15 / 32 * npix_x):int(17 / 32 * npix_x)]

        X_2 = X[::2, ::2]
        Y_2 = Y[::2, ::2]
        X_5 = X[::5, ::5]
        Y_5 = Y[::5, ::5]
        X_10 = X[::10, ::10]
        Y_10 = Y[::10, ::10]

        u = -1 * np.sin(EVPA_mask_circmean)
        v = np.cos(EVPA_mask_circmean)
        u_2 = u[::2, ::2]
        v_2 = v[::2, ::2]
        u_5 = u[::5, ::5]
        v_5 = v[::5, ::5]
        u_10 = u[::10, ::10]
        v_10 = v[::10, ::10]


        ### Plotting
        filename = f"{output_directory}/{output_directory}_MGVI_sky_compact_region_250muas.png"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,20))

        skyI = ax1.pcolormesh(X, Y, StokesI_sky, cmap=cm.inferno)
        ax1.set_aspect('equal')
        ax1.set_title('Stokes I', fontsize=20)
        cbar_ax1 = fig.colorbar(skyI, ax=ax1)
        cbar_ax1.set_label('Jy/str')

        frac_lin_pol = ax2.pcolormesh(X, Y, Frac_linear_pol_postmean, cmap=cm.gist_rainbow)
        I_contour_ax2 = ax2.contour(X, Y, StokesI_sky, 3, linewidths=0.3)
        ax2.set_aspect('equal')
        ax2.set_title('Fractional linear polarization', fontsize=20)
        cbar_ax2 = fig.colorbar(frac_lin_pol, ax=ax2)

        lin_pol = ax3.pcolormesh(X, Y, Linear_pol_postmean, cmap=cm.gist_rainbow)
        I_contour_ax3 = ax3.contour(X, Y, StokesI_sky, 3, linewidths=0.3)
        ax3.set_aspect('equal')
        ax3.set_title('Linear polarization', fontsize=20)
        cbar_ax3 = fig.colorbar(lin_pol, ax=ax3)
        cbar_ax3.set_label('Jy/str')

        q1 = ax4.quiver(X_2, Y_2, u_2, v_2, scale=7, scale_units='inches', width=0.001, headwidth=0, headlength=0, headaxislength=0)
        I_contour_ax4 = ax4.contour(X, Y, StokesI_sky, 3, linewidths=0.3)
        ax4.set_aspect('equal')
        ax4.set_title('EVPA', fontsize=20)
        #cbar_ax6 = fig.colorbar(sky_EVPA, ax=ax6)
        #cbar_ax6.remove()

        #skyV = ax5.pcolormesh(X, Y, StokesV_sky, cmap=cm.bwr)
        #ax5.set_aspect('equal')
        #ax5.set_title('Stokes V', fontsize=20)
        #cbar_ax5 = fig.colorbar(skyV, ax=ax5)
        #cbar_ax5.set_label('Jy/str')

        #sky_EVPA = ax6.pcolormesh(X, Y, EVPA, cmap=cm.hsv)
        ##I_contour_ax5 = ax5.contour(X, Y, StokesI_sky, levels=np.logspace(np.log10(15 * sigma), 2, num=3),linewidths=0.3)
        #ax6.set_aspect('equal')
        #ax6.set_title('EVPA', fontsize=20)
        #cbar_ax6 = fig.colorbar(sky_EVPA, ax=ax6)
        #cbar_ax6.set_label('deg')

        plt.savefig(filename)
#exit()



### Amplitude RCP gain plotter
with h5py.File(f"{output_directory}/logamp_RCP/last.hdf5", "r") as hdf:

    amp_samples_list = []

    for ii in range(len(hdf["samples"])):
        amp_sample = np.exp(hdf["samples"][f"{ii}"][()])
        amp_samples_list.append(amp_sample)

    amp_samples_mean = np.mean(amp_samples_list, axis=0)
    amp_samples_std = np.std(amp_samples_list, axis=0)

    N = amp_samples_mean.shape[1]
    T = solution_interval
    time_domain = np.linspace(0, N * T, N, endpoint=False)
    time_domain_hours = time_domain / 3600


    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]
    for jj in np.arange(total_number_of_spectral_window):
        fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))              
        filename = f"{output_directory}/{output_directory}_amp_gain_RCP_IF{jj}.png"

        for ii in np.arange(int(len(uantennas))):
            ax1.plot(time_domain_hours_half, amp_samples_mean[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2],
                    label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            ax1.fill_between(time_domain_hours_half,
                            amp_samples_mean[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] - amp_samples_std[total_number_of_spectral_window*ii+jj,
                                                                                :len(time_domain) // 2],
                            amp_samples_mean[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] + amp_samples_std[total_number_of_spectral_window*ii+jj,
                                                                                :len(time_domain) // 2],
                            alpha=0.5)
        ax1.set_title('Amplitude gain term RCP', fontsize=15)
        ax1.set_xlabel('hour')
        ax1.set_ylabel('amplitude gain')
        ax1.set_ylim(0.0, 2.0)
        ax1.legend(loc='upper right')
        plt.savefig(filename)
        print(f"Saved results as {filename}.")
        plt.close()


### Amplitude LCP gain plotter
with h5py.File(f"{output_directory}/logamp_LCP/last.hdf5", "r") as hdf:

    amp_samples_list = []

    for ii in range(len(hdf["samples"])):
        amp_sample = np.exp(hdf["samples"][f"{ii}"][()])
        amp_samples_list.append(amp_sample)

    amp_samples_mean = np.mean(amp_samples_list, axis=0)
    amp_samples_std = np.std(amp_samples_list, axis=0)

    N = amp_samples_mean.shape[1]
    T = solution_interval
    time_domain = np.linspace(0, N * T, N, endpoint=False)
    time_domain_hours = time_domain / 3600


    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]
    for jj in np.arange(total_number_of_spectral_window):
        fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))              
        filename = f"{output_directory}/{output_directory}_amp_gain_LCP_IF{jj}.png"

        for ii in np.arange(int(len(uantennas))):
            ax1.plot(time_domain_hours_half, amp_samples_mean[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2],
                    label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            ax1.fill_between(time_domain_hours_half,
                            amp_samples_mean[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] - amp_samples_std[total_number_of_spectral_window*ii+jj,
                                                                                :len(time_domain) // 2],
                            amp_samples_mean[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] + amp_samples_std[total_number_of_spectral_window*ii+jj,
                                                                                :len(time_domain) // 2],
                            alpha=0.5)
        ax1.set_title('Amplitude gain term LCP', fontsize=15)
        ax1.set_xlabel('hour')
        ax1.set_ylabel('amplitude gain')
        ax1.set_ylim(0.0, 2.0)
        ax1.legend(loc='upper right')
        plt.savefig(filename)
        print(f"Saved results as {filename}.")
        plt.close()




### Phase gain RCP plotter
with h5py.File(f"{output_directory}/phase_RCP/last.hdf5", "r") as hdf:

    phase_samples_list = []

    for ii in range(len(hdf["samples"])):
        phase_sample = hdf["samples"][f"{ii}"][()]
        phase_samples_list.append(phase_sample)

    phase_samples_mean = np.mean(phase_samples_list, axis=0)
    phase_samples_mean_deg = 180 / np.pi * phase_samples_mean

    phase_samples_std = np.std(phase_samples_list, axis=0)
    phase_samples_std_deg = 180 / np.pi * phase_samples_std

    phase_antenna_mean = np.mean(phase_samples_mean_deg, axis=1)
    phase_antenna_std = np.std(phase_samples_mean_deg, axis=1)
    #print(f"RCP phase antenna mean: {phase_antenna_mean}")
    #print(f"RCP phase antenna std: {phase_antenna_std}")

    N = phase_samples_mean.shape[1]
    T = solution_interval
    time_domain = np.linspace(0, N * T, N, endpoint=False)
    time_domain_hours = time_domain / 3600

    ### MGVI/geoVI Phase gain RCP Plotter
    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]
    for jj in np.arange(total_number_of_spectral_window):
        fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))              
        filename = f"{output_directory}/{output_directory}_phase_gain_RCP_IF{jj}.png"

        for ii in np.arange(int(len(uantennas))):
            ax1.plot(time_domain_hours_half, phase_samples_mean_deg[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2],
                    label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            ax1.fill_between(time_domain_hours_half,
                            phase_samples_mean_deg[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] - phase_samples_std_deg[total_number_of_spectral_window*ii+jj,
                                                                                        :len(time_domain) // 2],
                            phase_samples_mean_deg[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] + phase_samples_std_deg[total_number_of_spectral_window*ii+jj,
                                                                                        :len(time_domain) // 2],
                            alpha=0.5)
        ax1.set_title('Phase gain RCP', fontsize=15)
        ax1.set_xlabel('hour')
        ax1.set_ylabel('deg')
        ax1.set_ylim(-90, 90)
        ax1.legend()
        plt.savefig(filename)
        print(f"Saved results as {filename}.")
        plt.close()


### Phase gain LCP plotter
with h5py.File(f"{output_directory}/phase_LCP/last.hdf5", "r") as hdf:

    phase_samples_list = []

    for ii in range(len(hdf["samples"])):
        phase_sample = hdf["samples"][f"{ii}"][()]
        phase_samples_list.append(phase_sample)

    phase_samples_mean = np.mean(phase_samples_list, axis=0)
    phase_samples_mean_deg = 180 / np.pi * phase_samples_mean

    phase_samples_std = np.std(phase_samples_list, axis=0)
    phase_samples_std_deg = 180 / np.pi * phase_samples_std

    phase_antenna_mean = np.mean(phase_samples_mean_deg, axis=1)
    phase_antenna_std = np.std(phase_samples_mean_deg, axis=1)
    #print(f"LCP phase antenna mean: {phase_antenna_mean}")
    #print(f"LCP phase antenna std: {phase_antenna_std}")

    N = phase_samples_mean.shape[1]
    T = solution_interval
    time_domain = np.linspace(0, N * T, N, endpoint=False)
    time_domain_hours = time_domain / 3600

    ### MGVI/geoVI Phase gain RCP Plotter
    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]
    for jj in np.arange(total_number_of_spectral_window):
        fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))              
        filename = f"{output_directory}/{output_directory}_phase_gain_LCP_IF{jj}.png"

        for ii in np.arange(int(len(uantennas))):
            ax1.plot(time_domain_hours_half, phase_samples_mean_deg[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2],
                    label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
            ax1.fill_between(time_domain_hours_half,
                            phase_samples_mean_deg[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] - phase_samples_std_deg[total_number_of_spectral_window*ii+jj,
                                                                                        :len(time_domain) // 2],
                            phase_samples_mean_deg[total_number_of_spectral_window*ii+jj, :len(time_domain) // 2] + phase_samples_std_deg[total_number_of_spectral_window*ii+jj,
                                                                                        :len(time_domain) // 2],
                            alpha=0.5)
        ax1.set_title('Phase gain LCP', fontsize=15)
        ax1.set_xlabel('hour')
        ax1.set_ylabel('deg')
        ax1.set_ylim(-90, 90)
        ax1.legend()
        plt.savefig(filename)
        print(f"Saved results as {filename}.")
        plt.close()


### Constant Dterm plotter
def load_constant_dterm_samples(name):
    with h5py.File(f"{output_directory}/{name}/last.hdf5", "r") as hdf:
        samples = hdf["samples"]
        return np.stack([samples[f"{ii}"][()] for ii in range(len(samples))], axis=0)[:, :, 0]


Dterm_logamp_RCP_samples = load_constant_dterm_samples("Dterm_logamp_RCP")
Dterm_phase_RCP_samples = load_constant_dterm_samples("Dterm_phase_RCP")
Dterm_logamp_LCP_samples = load_constant_dterm_samples("Dterm_logamp_LCP")
Dterm_phase_LCP_samples = load_constant_dterm_samples("Dterm_phase_LCP")

Dterm_RCP_samples = np.exp(Dterm_logamp_RCP_samples + 1j * Dterm_phase_RCP_samples)
Dterm_LCP_samples = np.exp(Dterm_logamp_LCP_samples + 1j * Dterm_phase_LCP_samples)
Dterm_RCP_mean = np.mean(Dterm_RCP_samples, axis=0)
Dterm_LCP_mean = np.mean(Dterm_LCP_samples, axis=0)
#TODO check how to calculate std of complex number
Dterm_RCP_std = np.std(Dterm_RCP_samples, axis=0)
Dterm_LCP_std = np.std(Dterm_LCP_samples, axis=0)


def save_dterm_summary(filename):
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(
            "station\tantenna_id\tIF\tpolarization\treal\timag\t"
            "magnitude_percent\tstd_percent\n"
        )
        for polarization, means, stds in (
            ("RCP", Dterm_RCP_mean, Dterm_RCP_std),
            ("LCP", Dterm_LCP_mean, Dterm_LCP_std),
        ):
            for ii, antenna_id in enumerate(uantennas):
                station = station_table[antenna_id]
                for jj in range(means.shape[1]):
                    mean = means[ii, jj]
                    stream.write(
                        f"{station}\t{antenna_id}\t{jj}\t{polarization}\t"
                        f"{mean.real:.10e}\t{mean.imag:.10e}\t"
                        f"{100*abs(mean):.10e}\t{100*stds[ii, jj]:.10e}\n"
                    )


dterm_summary_filename = (
    f"{output_directory}/{output_directory}_Dterms.txt"
)
save_dterm_summary(dterm_summary_filename)
print(f"Saved results as {dterm_summary_filename}.")


for ii in range(len(uantennas)):
    station = station_table[list(uantennas)[ii]]
    for jj in range(Dterm_RCP_mean.shape[1]):
        mean = Dterm_RCP_mean[ii, jj]
        std = Dterm_RCP_std[ii, jj]
        print(
            f'{station} IF{jj} RCP Dterm : {100*abs(mean):.3f} '
            f'+- {100*std:.3f}% ({mean:.3f})'
        )
print()
for ii in range(len(uantennas)):
    station = station_table[list(uantennas)[ii]]
    for jj in range(Dterm_LCP_mean.shape[1]):
        mean = Dterm_LCP_mean[ii, jj]
        std = Dterm_LCP_std[ii, jj]
        print(
            f'{station} IF{jj} LCP Dterm : {100*abs(mean):.3f} '
            f'+- {100*std:.3f}% ({mean:.3f})'
        )


### Dterm Plotter
filename = f"{output_directory}/{output_directory}_Dterm_posterior_mean.png"
fig, ((ax1) ,(ax2)) = plt.subplots(2, 1, figsize=(6, 12))

for ii in np.arange(int(len(uantennas))):
    ax1.scatter(Dterm_RCP_mean.real[ii], Dterm_RCP_mean.imag[ii],
             label=f'{station_table[list(uantennas)[ii]]}')

ax1.set_title('RCP Dterms', fontsize=15)
ax1.set_xlabel('real')
ax1.set_ylabel('imag')
ax1.set_xlim(-0.2, 0.2)
ax1.set_ylim(-0.2, 0.2)
ax1.legend(loc='lower left')

for ii in np.arange(int(len(uantennas))):
    ax2.scatter(Dterm_LCP_mean.real[ii], Dterm_LCP_mean.imag[ii],
             label=f'{station_table[list(uantennas)[ii]]}')

ax2.set_title('LCP Dterms', fontsize=15)
ax2.set_xlabel('real')
ax2.set_ylabel('imag')
ax2.set_xlim(-0.2, 0.2)
ax2.set_ylim(-0.2, 0.2)
ax2.legend(loc='lower left')

plt.savefig(filename)
print(f"Saved results as {filename}.")
plt.close()

'''
### Dterm Plotter
filename = f"{output_directory}/{output_directory}_Dterm_posterior_mean_ehtim_comparison.png"
fig, ((ax1) ,(ax2)) = plt.subplots(2, 1, figsize=(6, 12))

for ii in np.arange(int(len(uantennas))):
    if ii==0 or ii ==2 or ii==3 or ii==4 or ii==5 or ii==6 or ii==10 or ii==11:
        ax1.scatter(Dterm_RCP_mean.real[ii], Dterm_RCP_mean.imag[ii],
                 label=f'{station_table[list(uantennas)[ii]]}')

ax1.set_title('RCP Dterms', fontsize=15)
ax1.set_xlabel('real')
ax1.set_ylabel('imag')
ax1.set_xlim(-0.2, 0.2)
ax1.set_ylim(-0.2, 0.2)
ax1.legend(loc='lower left')

for ii in np.arange(int(len(uantennas))):
    if ii==0 or ii ==2 or ii==3 or ii==4 or ii==5 or ii==6 or ii==10 or ii==11:
        ax2.scatter(Dterm_LCP_mean.real[ii], Dterm_LCP_mean.imag[ii],
                 label=f'{station_table[list(uantennas)[ii]]}')

ax2.set_title('LCP Dterms', fontsize=15)
ax2.set_xlabel('real')
ax2.set_ylabel('imag')
ax2.set_xlim(-0.2, 0.2)
ax2.set_ylim(-0.2, 0.2)
ax2.legend(loc='lower left')

plt.savefig(filename)
print(f"Saved results as {filename}.")
plt.close()
'''
exit()

### Time dependent Dterm plotter
### Amplitude Dterm plotter
with h5py.File(f"{output_directory}/Dterm_logamp/last.hdf5", "r") as hdf:

    amp_samples_list = []

    for ii in range(len(hdf["samples"])):
        amp_sample = np.exp(hdf["samples"][f"{ii}"][()])
        amp_samples_list.append(amp_sample)

    amp_samples_mean = np.mean(amp_samples_list, axis=0)
    amp_samples_std = np.std(amp_samples_list, axis=0)


    N = amp_samples_mean.shape[1]
    T = 10
    time_domain = np.linspace(0, N * T, N, endpoint=False)
    time_domain_hours = time_domain / 3600

    ### Amplitude Dterm RCP Plotter
    filename = f"{output_directory}/{output_directory}_amp_Dterm_RCP.png"
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))

    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

    for ii in np.arange(int(len(uantennas))):
        ax1.plot(time_domain_hours_half, amp_samples_mean[ii, :len(time_domain) // 2],
                 label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
        ax1.fill_between(time_domain_hours_half,
                         amp_samples_mean[ii, :len(time_domain) // 2] - amp_samples_std[ii,
                                                                              :len(time_domain) // 2],
                         amp_samples_mean[ii, :len(time_domain) // 2] + amp_samples_std[ii,
                                                                              :len(time_domain) // 2],
                         alpha=0.5)
    ax1.set_title('Amplitude Dterm RCP', fontsize=15)
    ax1.set_xlabel('hour')
    ax1.set_ylabel('amplitude Dterm')
    ax1.set_ylim(0.0, 0.2)
    ax1.legend(loc='upper right')
    plt.savefig(filename)
    print(f"Saved results as {filename}.")
    plt.close()

    ### Amplitude Dterm LCP Plotter
    filename = f"{output_directory}/{output_directory}_amp_Dterm_LCP.png"
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))

    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

    for ii in np.arange(int(len(uantennas)), 2*int(len(uantennas))):
        ax1.plot(time_domain_hours_half, amp_samples_mean[ii, :len(time_domain) // 2],
                 label=f'{station_table[list(uantennas)[ii - int(len(uantennas))]]}', linewidth=1.0)
        ax1.fill_between(time_domain_hours_half,
                         amp_samples_mean[ii, :len(time_domain) // 2] - amp_samples_std[ii,
                                                                              :len(time_domain) // 2],
                         amp_samples_mean[ii, :len(time_domain) // 2] + amp_samples_std[ii,
                                                                              :len(time_domain) // 2],
                         alpha=0.5)
    ax1.set_title('Amplitude Dterm LCP', fontsize=15)
    ax1.set_xlabel('hour')
    ax1.set_ylabel('amplitude Dterm')
    ax1.set_ylim(0.0, 0.2)
    ax1.legend(loc='upper right')
    plt.savefig(filename)
    print(f"Saved results as {filename}.")
    plt.close()


### Phase Dterm plotter
with h5py.File(f"{output_directory}/Dterm_phase/last.hdf5", "r") as hdf:

    phase_samples_list = []

    for ii in range(len(hdf["samples"])):
        phase_sample = hdf["samples"][f"{ii}"][()]
        phase_samples_list.append(phase_sample)

    phase_samples_mean = np.mean(phase_samples_list, axis=0)
    phase_samples_mean_deg = 180 / np.pi * phase_samples_mean

    phase_samples_std = np.std(phase_samples_list, axis=0)
    phase_samples_std_deg = 180 / np.pi * phase_samples_std

    N = phase_samples_mean.shape[1]
    T = 10
    time_domain = np.linspace(0, N * T, N, endpoint=False)
    time_domain_hours = time_domain / 3600

    ### MGVI/geoVI Phase Dterm RCP and LCP Plotter
    filename = f"{output_directory}/{output_directory}_phase_Dterm_RCP.png"
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))
    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

    for ii in np.arange(int(len(uantennas))):
        ax1.plot(time_domain_hours_half, phase_samples_mean_deg[ii, :len(time_domain) // 2],
                 label=f'{station_table[list(uantennas)[ii]]}', linewidth=1.0)
        ax1.fill_between(time_domain_hours_half,
                         phase_samples_mean_deg[ii, :len(time_domain) // 2] - phase_samples_std_deg[ii,
                                                                                    :len(time_domain) // 2],
                         phase_samples_mean_deg[ii, :len(time_domain) // 2] + phase_samples_std_deg[ii,
                                                                                    :len(time_domain) // 2],
                         alpha=0.5)
    ax1.set_title('Phase Dterm RCP', fontsize=15)
    ax1.set_xlabel('hour')
    ax1.set_ylabel('deg')
    ax1.set_ylim(-90, 90)
    ax1.legend()
    plt.savefig(filename)
    print(f"Saved results as {filename}.")
    plt.close()

    filename = f"{output_directory}/{output_directory}_phase_Dterm_LCP.png"
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 12))
    time_domain_hours_half = time_domain_hours[:len(time_domain_hours) // 2]

    for ii in np.arange(int(len(uantennas)), 2 * int(len(uantennas))):
        ax1.plot(time_domain_hours_half, phase_samples_mean_deg[ii, :len(time_domain) // 2],
                 label=f'{station_table[list(uantennas)[ii - int(len(uantennas))]]}', linewidth=1.0)
        ax1.fill_between(time_domain_hours_half,
                         phase_samples_mean_deg[ii, :len(time_domain) // 2] - phase_samples_std_deg[ii,
                                                                                    :len(time_domain) // 2],
                         phase_samples_mean_deg[ii, :len(time_domain) // 2] + phase_samples_std_deg[ii,
                                                                                    :len(time_domain) // 2],
                         alpha=0.5)
    ax1.set_title('Phase Dterm LCP', fontsize=15)
    ax1.set_xlabel('hour')
    ax1.set_ylabel('deg')
    ax1.set_ylim(-90, 90)
    ax1.legend()
    plt.savefig(filename)
    print(f"Saved results as {filename}.")
    plt.close()
