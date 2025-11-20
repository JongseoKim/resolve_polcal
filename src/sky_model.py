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

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from resolve.constants import str2rad
from resolve.irg_space import IRGSpace
from resolve.polarization_space import PolarizationSpace
from resolve.polarization_matrix_exponential import polarization_matrix_exponential_mf2f
from resolve.sky_model import default_sky_domain
from resolve.util import assert_sky_domain, my_assert
from resolve.integrated_wiener_process import  _FancyBroadcast
from .ift_cfm_maker import cfm_from_cfg




def sky_model_diffuse(cfg, source_number=0, observations=[], nthreads=1):
    sdom = _spatial_dom(cfg)
    pdom = PolarizationSpace(cfg["polarization"].split(","))

    additional = {}
    logsky = {}

    for lbl in pdom.labels:
        pol_lbl = f"{lbl.upper()}"
        if cfg["freq mode"] == "single":
            op, aa = _single_freq_logsky(cfg, pol_lbl, source_number)
        elif cfg["freq mode"] == "cfm":
            op, aa = _multi_freq_logsky_cfm(cfg, sdom, pol_lbl)
        elif cfg["freq mode"] == "quad":
            freq = _get_frequencies(cfg, observations)
            op, aa = _multi_freq_logsky_quad(cfg, sdom, pol_lbl, freq)
        else:
            raise RuntimeError
        logsky[lbl] = op
        additional = {**additional, **aa}
    if cfg["freq mode"] == "single":
        tgt = default_sky_domain(pdom=pdom, sdom=sdom)
    else:
        fdom = op.target[0]
        tgt = default_sky_domain(pdom=pdom, fdom=fdom, sdom=sdom)

    logsky = reduce(add, (oo.ducktape_left(lbl) for lbl, oo in logsky.items()))
    mexp = polarization_matrix_exponential_mf2f(logsky.target, nthreads=nthreads)
    sky = mexp @ logsky

    sky = sky.ducktape_left(tgt)
    assert_sky_domain(sky.target)

    return sky, logsky, additional


def _single_freq_logsky(cfg, pol_label, source_number=0):
    sdom = _spatial_dom(cfg)
    my_assert(type(source_number) == int)
    if source_number==0:
        cfm = cfm_from_cfg(cfg, {"space i0": sdom}, f"stokes{pol_label} diffuse")
    if source_number > 0:
        cfm = cfm_from_cfg(cfg, {"space i0": sdom}, f"stokes{pol_label} diffuse", domain_prefix=f"source{source_number} stokes{pol_label} diffuse ")
    op = cfm.finalize(0)
    additional = {
        f"logdiffuse stokes{pol_label} power spectrum": cfm.power_spectrum,
        f"logdiffuse stokes{pol_label}": op,
    }

    return op, additional

def _multi_freq_logsky_quad(cfg, sdom, pol_label, freq):
    assert len(freq) > 0

    fdom = IRGSpace(freq)

    prefix = f"stokes{pol_label} diffuse"
    i0_cfm = cfm_from_cfg(cfg, {"space i0": sdom}, f"stokes{pol_label} diffuse")
    alpha_cfm = cfm_from_cfg(cfg, {"": sdom}, prefix + " space alpha")
    beta_cfm = cfm_from_cfg(cfg, {"": sdom}, prefix + " space beta")
    log_i0 = i0_cfm.finalize(0)
    alpha = alpha_cfm.finalize(0)
    beta = beta_cfm.finalize(0)

    log_fdom = IRGSpace(np.log(freq / freq.mean()))


    additional = {
        f"stokes{pol_label} i0": log_i0,
        f"stokes{pol_label} alpha": alpha,
        f"stokes{pol_label} beta": beta,
    }


    tgt = log_fdom, log_i0.target[0]
    sdom_ = tgt[0]

    bc = _FancyBroadcast(tgt)
    factors = ift.full(sdom_, 0)
    factors = np.empty(sdom_.shape)
    factors[0] = 0
    factors[1:] = np.cumsum(sdom_.distances)
    factors1 = ift.makeField(sdom_, factors)
    factors2 = ift.makeField(sdom_, factors**2)
    op_quad = bc @ log_i0 + ift.DiagonalOperator(factors1, tgt, 0) @ bc @ alpha + ift.DiagonalOperator(factors2, tgt, 0) @ bc @ beta
    op_quad = op_quad.ducktape_left((fdom, sdom))

    return op_quad, additional



def _multi_freq_logsky_cfm(cfg, sdom, pol_label):
    fnpix, df = cfg.getfloat("freq npix"), cfg.getfloat("freq pixel size")
    freq0 = cfg.getfloat("freq start")

    if fnpix is None:
        raise ValueError("Please set a value for 'freq npix'")
    if df is None:
        raise ValueError("Please set a value for 'freq pixel size'")
    if freq0 is None:
        raise ValueError("Please set a value for 'freq start'")

    fdom = IRGSpace(freq0 + np.arange(fnpix)*df)
    fdom_rg = ift.RGSpace(fnpix,df)


    cfm = cfm_from_cfg(cfg, {"freq": fdom_rg, "space i0": sdom}, f"stokes{pol_label} diffuse")
    op = cfm.finalize(0)
    additional = {}

    return op.ducktape_left((fdom,sdom)), additional



def _spatial_dom(cfg):
    nx = cfg.getint("space npix x")
    ny = cfg.getint("space npix y")
    dx = str2rad(cfg["space fov x"]) / nx
    dy = str2rad(cfg["space fov y"]) / ny
    return ift.RGSpace([nx, ny], [dx, dy])


def _get_frequencies(cfg, observations):
    if cfg["frequencies"] == "data":
        freq = np.array([oo.freq for oo in observations]).flatten()
    else:
        freq = map(float, cfg["frequencies"].split(","))
    return np.sort(np.array(list(freq)))