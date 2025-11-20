import numpy as np
from nifty8 import ScalingOperator, makeOp, GaussianEnergy, ContractionOperator, DomainChangerAndReshaper, makeOp, Field, MultiDomain, MultiField, LognormalTransform, DomainTuple, UnstructuredDomain, Realizer, makeField, makeDomain, FieldAdapter
from resolve.response import InterferometryResponse
from resolve.dtype_converter import DtypeConverter
from resolve.calibration import CalibrationDistributor
from resolve.data.observation import unique_antennas
from resolve.polarization_space import PolarizationSpace
from src.utilities import Expander_R000, Expander_0R00, Expander_00R0, Expander_000R

def likelihood_calib_logamplitudes(obs):
    vis = obs.vis.ducktape_left(f"vis")
    weight = obs.weight.ducktape_left(f"vis")
    snr = makeOp(vis.abs() * weight.sqrt())
    inp_vis = ScalingOperator(vis.domain, 1.0)
    inp_cal = (
        ScalingOperator(obs.vis.domain, 1.0)
        .ducktape(f"cal")
        .ducktape_left(f"vis")
    )
    vis2calibnormlogampl = snr @ (inp_vis.log().real + inp_cal)
    vis2normlogampl = snr @ inp_vis.log().real

    lh_calibrated_amplitude = GaussianEnergy(data=vis2normlogampl(vis)) @ vis2calibnormlogampl

    return lh_calibrated_amplitude


def lhmix(
        obs,
        sky_model,
        calibration_model,
        mix_scale,
        lh_closure_phases,
        lh_closure_amplitudes,
        lh_calibration_logamplitudes
):

    lh1 = mix_scale * lh_closure_amplitudes + lh_closure_phases
    lh2 = (1 - mix_scale) * lh_calibration_logamplitudes

    internal_sky_key = "_sky"
    R = InterferometryResponse(
        obs,
        sky_model.target,
        do_wgridding=False,
        epsilon=1e-6
    ).ducktape(internal_sky_key).ducktape_left(f"vis")
    signal_response1 = R

    #TODO check sanity
    if calibration_model != "None":
        signal_response2 = signal_response1 + calibration_model
    else:
        signal_response2 = signal_response1

    lh_mix = lh1 @ signal_response1 + lh2 @ signal_response2
    sky_model = sky_model.ducktape_left(internal_sky_key)
    return lh_mix.partial_insert(sky_model)


def likelihood_pol(
        obs,
        obs_RR,
        sky_model,
        pol_cal_op,
        verbosity=0
):

    internal_sky_key = "_sky"
    R = InterferometryResponse(
        obs,
        sky_model.target,
        do_wgridding=False,
        epsilon=1e-6,
        verbosity=verbosity
    ).ducktape(internal_sky_key)

    dtype = obs.vis.dtype
    dt = DtypeConverter(pol_cal_op.target, np.complex128, dtype)
    R = (dt @ pol_cal_op) * R

    ContractionOp = ContractionOperator(obs.vis.domain, spaces=0)
    DomainChangerOp = DomainChangerAndReshaper(ContractionOp.target,
                                               [obs_RR.domain[0], obs.vis.domain[1], obs.vis.domain[2]])
    DomConOp = DomainChangerOp @ ContractionOp

    likelihood_ = GaussianEnergy(obs_RR) @ DomConOp @ R
    sky_operator = sky_model.ducktape_left(internal_sky_key)

    return likelihood_.partial_insert(sky_operator)



def PolarizationLikelihood(
        obs,
        R,
        sky,
        pol_cal_op_RR,
        pol_cal_op_RL,
        pol_cal_op_LR,
        pol_cal_op_LL,
):

    ContractionOp = ContractionOperator(obs.vis.domain, spaces=0)
    ExpanderOp_R000 = Expander_R000(ContractionOp.target)
    ExpanderOp_0R00 = Expander_0R00(ContractionOp.target)
    ExpanderOp_00R0 = Expander_00R0(ContractionOp.target)
    ExpanderOp_000R = Expander_000R(ContractionOp.target)


    Expander_ContractionOp_RR = ExpanderOp_R000 @ ContractionOp
    Expander_ContractionOp_RL = ExpanderOp_0R00 @ ContractionOp
    Expander_ContractionOp_LR = ExpanderOp_00R0 @ ContractionOp
    Expander_ContractionOp_LL = ExpanderOp_000R @ ContractionOp

    Rs = R @ sky
    Rs = Rs.ducktape_left("Rs")

    pol_cal_op_RR = pol_cal_op_RR.ducktape_left("RR")
    pol_cal_op_RL = pol_cal_op_RL.ducktape_left("RL")
    pol_cal_op_LR = pol_cal_op_LR.ducktape_left("LR")
    pol_cal_op_LL = pol_cal_op_LL.ducktape_left("LL")

    Rs_calib = Rs + pol_cal_op_RR + pol_cal_op_RL + pol_cal_op_LR + pol_cal_op_LL
    Rs_input = FieldAdapter(R.target, "Rs")
    pol_RR_input = FieldAdapter(R.target, "RR")
    pol_RL_input = FieldAdapter(R.target, "RL")
    pol_LR_input = FieldAdapter(R.target, "LR")
    pol_LL_input = FieldAdapter(R.target, "LL")


    #Rs_polcal = Expander_ContractionOp_RR @ (pol_cal_op_RR * Rs) + Expander_ContractionOp_RL @ (pol_cal_op_RL * Rs) + Expander_ContractionOp_LR @ (pol_cal_op_LR * Rs) + Expander_ContractionOp_LL @ (pol_cal_op_LL * Rs)
    Rs_polcal = Expander_ContractionOp_RR @ (pol_RR_input * Rs_input) + Expander_ContractionOp_RL @ (pol_RL_input * Rs_input) + Expander_ContractionOp_LR @ (pol_LR_input * Rs_input) + Expander_ContractionOp_LL @ (pol_LL_input * Rs_input)
    Rs_polcal = Rs_polcal @ Rs_calib
    weight_op = makeOp(obs.weight, sampling_dtype=obs.vis.dtype)

    Likelihood = GaussianEnergy(data=obs.vis, inverse_covariance=weight_op) @ Rs_polcal

    return Likelihood, Rs_polcal


def PolCalibratorLikelihood(
        obs,
        cal_total_intensity,
        cal_frac_linpol,
        cal_pol_angle_rad,
        pol_cal_op_RR,
        pol_cal_op_RL,
        pol_cal_op_LR,
        pol_cal_op_LL,
):

    ContractionOp = ContractionOperator(obs.vis.domain, spaces=0)
    ExpanderOp_R000 = Expander_R000(ContractionOp.target)
    ExpanderOp_0R00 = Expander_0R00(ContractionOp.target)
    ExpanderOp_00R0 = Expander_00R0(ContractionOp.target)
    ExpanderOp_000R = Expander_000R(ContractionOp.target)


    Expander_ContractionOp_RR = ExpanderOp_R000 @ ContractionOp
    Expander_ContractionOp_RL = ExpanderOp_0R00 @ ContractionOp
    Expander_ContractionOp_LR = ExpanderOp_00R0 @ ContractionOp
    Expander_ContractionOp_LL = ExpanderOp_000R @ ContractionOp

    Rs_calmodel = np.zeros(obs.vis.domain.shape, dtype=complex)
    I_modelvis = cal_total_intensity + 0j
    Q_modelvis = cal_total_intensity * cal_frac_linpol * np.cos(2 * cal_pol_angle_rad) + 0j
    U_modelvis = cal_total_intensity * cal_frac_linpol * np.sin(2 * cal_pol_angle_rad) + 0j
    V_modelvis = 0 + 0j

    vis_array = np.array([I_modelvis+V_modelvis, Q_modelvis+1j*U_modelvis, Q_modelvis-1j*U_modelvis, I_modelvis-V_modelvis])

    for ii in range(4):
        for jj in range(obs.vis.domain.shape[1]):
            Rs_calmodel[ii, jj, 0] = vis_array[ii]

    Rs_calmodel_field = Field.from_raw(obs.vis.domain, Rs_calmodel)
    Rs_cal = makeOp(inp=Rs_calmodel_field, dom=obs.vis.domain, sampling_dtype=obs.vis.dtype)
    Rs_polcal =  Expander_ContractionOp_RR @ (Rs_cal @ pol_cal_op_RR) + Expander_ContractionOp_RL @ (Rs_cal @ pol_cal_op_RL) + Expander_ContractionOp_LR @ (Rs_cal @ pol_cal_op_LR) + Expander_ContractionOp_LL @ (Rs_cal @ pol_cal_op_LL)

    weight_op = makeOp(obs.weight, sampling_dtype=obs.vis.dtype)
    Likelihood = GaussianEnergy(data=obs.vis, inverse_covariance=weight_op) @ Rs_polcal

    return Likelihood, Rs_calmodel_field, Rs_polcal


def CalibratorLikelihood(
        obs_cal,
        gain_logamp_op,
        gain_phase_op,
        flux_cal_total_intensity = 1.0,
        gain_calibrator=False,
        single_gain=False,
        gain_cal_flux_mean=1.0,
        gain_cal_flux_std=0.5,
):
    if gain_calibrator==False:
        Rs_flux_calmodel = np.zeros(obs_cal.vis.domain.shape, dtype=complex)
        LL_vis = flux_cal_total_intensity + 0j
        RR_vis = flux_cal_total_intensity + 0j

        vis_array = np.array([LL_vis, RR_vis])

        for ii in range(2):
            for jj in range(obs_cal.vis.domain.shape[1]):
                Rs_flux_calmodel[ii, jj, 0] = vis_array[ii]

        Rs_flux_calmodel_field = Field.from_raw(obs_cal.vis.domain, Rs_flux_calmodel)
        Rs_cal = makeOp(inp=Rs_flux_calmodel_field, dom=obs_cal.vis.domain,
                                 sampling_dtype=obs_cal.vis.dtype)

    else:
        key = 'gain_calibrator_flux'
        Rs_cal___ = LognormalTransform(gain_cal_flux_mean, gain_cal_flux_std, key, N_copies=1)
        Complexifier = Realizer(Rs_cal___.target).adjoint
        Rs_cal__ = Complexifier @ Rs_cal___
        if single_gain==False:
            target = DomainTuple.make([UnstructuredDomain(2*obs_cal.vis.domain.shape[1]),
                                   UnstructuredDomain(obs_cal.vis.domain.shape[2])])
        else:
            target = DomainTuple.make([UnstructuredDomain(obs_cal.vis.domain.shape[1]),
                                       UnstructuredDomain(obs_cal.vis.domain.shape[2])])

        Expander = ContractionOperator(target, spaces=0).adjoint
        Rs_cal_ = Expander @ Rs_cal__
        DomainChanger = DomainChangerAndReshaper(Rs_cal_.target, obs_cal.vis.domain)
        Rs_cal = DomainChanger @ Rs_cal_

        pdom, _, fdom = obs_cal.vis.domain
        reshaper = DomainChangerAndReshaper(
            gain_logamp_op.target,
            (
                pdom,
                gain_logamp_op.target[0],
                gain_logamp_op.target[1],
                fdom
            )
        )
        gain_logamp_op = reshaper @ gain_logamp_op
        gain_phase_op = reshaper @ gain_phase_op

    uantennas = unique_antennas(obs_cal)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    cop1 = CalibrationDistributor(gain_logamp_op.target, Rs_cal.target, obs_cal.ant1, obs_cal.time, antenna_dct, None)
    cop2 = CalibrationDistributor(gain_logamp_op.target, Rs_cal.target, obs_cal.ant2, obs_cal.time, antenna_dct, None)
    res0 = (cop1 + cop2).real @ gain_logamp_op
    res1 = (1j * (cop1 - cop2).real) @ gain_phase_op
    gigj_op = (res0 + res1).exp()

    if gain_calibrator == False:
        gigj_Rs_cal = Rs_cal @ gigj_op
    else:
        gigj_Rs_cal = Rs_cal * gigj_op

    weight_op = makeOp(obs_cal.weight, sampling_dtype=obs_cal.vis.dtype)

    Likelihood = GaussianEnergy(data=obs_cal.vis, inverse_covariance=weight_op) @ gigj_Rs_cal

    return Likelihood, gigj_Rs_cal


def PolarizationLikelihood_combined(
        obs,
        obs_RR,
        obs_RL,
        obs_LR,
        obs_LL,
        sky_model,
        Response,
        pol_cal_op_RR,
        pol_cal_op_RL,
        pol_cal_op_LR,
        pol_cal_op_LL,
):

    internal_sky_key = "_sky"
    R = Response.ducktape(internal_sky_key)

    dtype = obs.vis.dtype
    dt = DtypeConverter(pol_cal_op_RR.target, np.complex128, dtype)

    ContractionOp = ContractionOperator(obs.vis.domain, spaces=0)
    DomainChangerOp1 = DomainChangerAndReshaper(ContractionOp.target,
                                               [obs_RR.domain[0], obs.vis.domain[1], obs.vis.domain[2]])
    DomConOp1 = DomainChangerOp1 @ ContractionOp
    DomainChangerOp2 = DomainChangerAndReshaper(ContractionOp.target,
                                               [obs_RL.domain[0], obs.vis.domain[1], obs.vis.domain[2]])
    DomConOp2 = DomainChangerOp2 @ ContractionOp
    DomainChangerOp3 = DomainChangerAndReshaper(ContractionOp.target,
                                               [obs_LR.domain[0], obs.vis.domain[1], obs.vis.domain[2]])
    DomConOp3 = DomainChangerOp3 @ ContractionOp
    DomainChangerOp4 = DomainChangerAndReshaper(ContractionOp.target,
                                               [obs_LL.domain[0], obs.vis.domain[1], obs.vis.domain[2]])
    DomConOp4 = DomainChangerOp4 @ ContractionOp

    sky_operator = sky_model.ducktape_left(internal_sky_key)
    likelihood_RR = GaussianEnergy(obs_RR) @ DomConOp1 @ ((dt @ pol_cal_op_RR) * R).partial_insert(sky_operator)
    likelihood_RL = GaussianEnergy(obs_RL) @ DomConOp2 @ ((dt @ pol_cal_op_RL) * R).partial_insert(sky_operator)
    likelihood_LR = GaussianEnergy(obs_LR) @ DomConOp3 @ ((dt @ pol_cal_op_LR) * R).partial_insert(sky_operator)
    likelihood_LL = GaussianEnergy(obs_LL) @ DomConOp4 @ ((dt @ pol_cal_op_LL) * R).partial_insert(sky_operator)


    return likelihood_RR + likelihood_RL + likelihood_LR + likelihood_LL