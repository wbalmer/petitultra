#!/usr/bin/env python
# coding: utf-8

# low level imports
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1" # stop e.g. numpy from doing own parallel with mpi
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # keep hdf5 file locking up across processes 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL) # for rockfish dlopen error

# basic imports
import time
import copy
import numpy as np
from astropy.io import fits
import multiprocess as mp
import warnings
import corner
import astropy.units as u
from scipy.stats import norm, truncnorm, loguniform
import matplotlib.pyplot as plt
from spectres import spectres
from species import SpeciesInit
from species.phot.syn_phot import SyntheticPhotometry

# sampler imports 
import ultranest
import ultranest.stepsampler
from ultranest.plot import cornerplot, runplot, traceplot, PredictionBand

# mpi imports
from mpi4py.futures import MPIPoolExecutor
from schwimmbad import MPIPool
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# prt specific imports
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global, dtdp_temperature_profile
from petitRADTRANS.radtrans import Radtrans # <--- this is our "spectrum generator," e.g. the radiative transfer solver
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.clouds import return_cloud_mass_fraction, simple_cdf
from petitRADTRANS.math import filter_spectrum_with_spline

from petitRADTRANS.retrieval.utils import log_prior, uniform_prior, gaussian_prior, log_gaussian_prior, delta_prior, inverse_gamma_prior

# this module imports
from atmo_funcs import *

# general setup
retrieval_name = '2M0624_SiO2NUC'
output_dir = retrieval_name+'_outputs/'
checkpoint_file = output_dir+f'checkpoint_{retrieval_name}.hdf5'

# sampling parameters
dlogz = 0.5
nlive = 400
resume = True
plot = True

from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # load data, start species if needed for photometry
    # time.sleep(rank) # sleeping here to avoid two processes trying to write to species.ini at the same time
    # SpeciesInit()

    data_path = './data/'
    data = np.loadtxt(data_path+'2M0624_bin300.txt')
    
    w = data[:,0]
    f = data[:,1]
    fe = data[:,2]

    unit_conv = (u.erg/u.s/u.cm**2/u.cm).to(u.W/u.m**2/u.micron) # units of pRT models to units of data

    chem = PreCalculatedEquilibriumChemistryTable()


    rtpressures = np.logspace(-6, 3, 100) # set pressure range
    line_species = [
        'H2O',
        'CO-NatAbund',
        'CH4',
        'CO2',
        'HCN',
        'FeH',
        'H2S',
        'NH3',
        'PH3',
        'Na',
        'K',
        'TiO',
        'VO',
        'SiO'
    ]
    rayleigh_species = ['H2', 'He'] # why is the sky blue?
    gas_continuum_contributors = ['H2--H2', 'H2--He'] # these are important sources of opacity
    cloud_species = [
                    # 'SiO2(s)_amorphous__DHS',
                    'SiO2(s)_amorphous__Mie',
                    # 'SiO(s)_amorphous__DHS',
                    # 'SiO(s)_amorphous__Mie',
                    'MgSiO3(s)_amorphous__DHS',
                    # 'MgSiO3(s)_amorphous__DHS',
                    'Fe(s)_amorphous__DHS'
                    ] # clouds

    nsmresl = '300' # model resolution, R=1000 c-k
    atmosphere = Radtrans(
        pressures = rtpressures,
        line_species = [i+f'.R{nsmresl}' for i in line_species],
        rayleigh_species = rayleigh_species, # why is the sky blue?
        gas_continuum_contributors = gas_continuum_contributors, # these are important sources of opacity
        cloud_species = cloud_species, # these will be important for clouds
        wavelength_boundaries = [w[0]-0.1, w[-1]+0.1],
        line_opacity_mode='c-k' # lbl or c-k
    )

    def spectrum_generator(params, return_extras=False, quench_co2_off_co=True, quench_nitrogen=True):
        planet_radius = params['R_pl']* cst.r_jup_mean
        parallax = params['plx']
        r2d2 = (planet_radius/(cst.pc/(parallax/1000)))**2
        if 'mass' in params.keys():
            reference_gravity = (cst.G*params['mass']*cst.m_jup)/(planet_radius**2)
        else:
            reference_gravity = 1e1**params['logg']
        
        pressures = atmosphere.pressures * 1e-6 # cgs to bar

        # gradient P-T profile from Zhang+23
        t_bottom = params['T_bottom']
        num_layer = 6 # 10 # params['N_layers']
        layer_pt_slopes = np.ones(num_layer) * np.nan
        for index in range(num_layer):
            layer_pt_slopes[index] = params[f'dPT_{num_layer - index}']
        temperature = dtdp_temperature_profile(
            pressures,
            num_layer,
            layer_pt_slopes,
            t_bottom,
            top_of_atmosphere_pressure=-6,
            bottom_of_atmosphere_pressure=3
        )

        co_ratio = params['C/O']
        feh = params['Fe/H']
        if "log_kzz_lower" in params.keys():
            log_kzz_lower = params["log_kzz_lower"]
            log_kzz_upper = params["log_kzz_upper"]
            p_kzz_trans = 10**params["log_p_kzz_trans"]
            log_kzz_chem = np.ones_like(pressures) * log_kzz_lower
            log_kzz_chem[pressures < p_kzz_trans] = log_kzz_upper
        else:
            log_kzz_chem = np.ones_like(pressures) * params['log_kzz_chem']

        co_ratios = co_ratio * np.ones_like(pressures)
        log10_metallicities = feh * np.ones_like(pressures)

        mmw_init = np.ones_like(pressures) * 2.33

        p_quench = kzz_to_co_pquench(temperature, pressures, mmw_init, reference_gravity, log_kzz_chem, log10_metallicities)

        mass_fractions, mean_molar_masses, nabla_ad = chem.interpolate_mass_fractions(
            co_ratios=co_ratios,
            log10_metallicities=log10_metallicities,
            temperatures=temperature,
            pressures=pressures,
            carbon_pressure_quench=p_quench,
            full=True
        )

        if quench_co2_off_co:
            if p_quench is not None:
                p_quench_co2 = kzz_to_co2_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem, log10_metallicities)
                # in between p_quench_co and p_quench_co2, use Keq, then past p_quench_co2, fix co2
                if p_quench_co2 is not None:
                    quenchish_idx = np.logical_and(pressures <= p_quench, pressures <= p_quench_co2)
                else:
                    quenchish_idx = pressures <= p_quench
                mf_h2 = mass_fractions['H2'][quenchish_idx]
                mf_co = mass_fractions['CO'][quenchish_idx]
                mf_h2o = mass_fractions['H2O'][quenchish_idx]
                Keq = 18.3*np.exp((-2376/temperature[quenchish_idx]) - ((932/temperature[quenchish_idx])**2))
                mass_fractions['CO2'][quenchish_idx] = (mf_co * mf_h2o)/(mf_h2 * Keq)
                if p_quench_co2 is not None:
                    quench_idx = np.min(
                        (
                            np.searchsorted(pressures, p_quench_co2),
                            pressures.size - 1
                        )
                    )
                    mass_fractions['CO2'][pressures < p_quench_co2] = \
                        mass_fractions['CO2'][quench_idx]

        if quench_nitrogen:
            p_quench_nh3 = kzz_to_nh3_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem)
            p_quench_hcn = kzz_to_hcn_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem, log10_metallicities)

            if p_quench_nh3 is not None:
                quench_idx = np.min(
                    (
                        np.searchsorted(pressures, p_quench_nh3),
                        pressures.size - 1
                    )
                )
                mass_fractions['NH3'][pressures < p_quench_nh3] = \
                    mass_fractions['NH3'][quench_idx]

            if p_quench_hcn is not None:
                quench_idx = np.min(
                    (
                        np.searchsorted(pressures, p_quench_hcn),
                        pressures.size - 1
                    )
                )
                mass_fractions['HCN'][pressures < p_quench_hcn] = \
                    mass_fractions['HCN'][quench_idx]



        # debug abund plot
        # plt.figure()
        # plt.plot(mass_fractions['H2O'],pressures, label="H2O")
        # plt.plot(mass_fractions['CO'], pressures, label="CO")
        # plt.plot(mass_fractions['CH4'], pressures, label="CH4")
        # plt.plot(mass_fractions['CO2'], pressures, label="CO2")
        # plt.plot(mass_fractions['NH3'], pressures, label="NH3")
        # plt.plot(mass_fractions['HCN'], pressures, label="HCN")
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.xlim(1e-8, 1e-1)
        # plt.ylim(1e3, 1e-6)
        # plt.hlines(p_quench, 1e-8,1e-1, color='k', ls='-', label="CO pq")
        # plt.hlines(p_quench_co2, 1e-8,1e-1, color='k', ls='--', label="CO2 pq")
        # plt.hlines(p_quench_hcn, 1e-8,1e-1, color='k', ls='-.', label="HCN pq")
        # plt.hlines(p_quench_nh3, 1e-8,1e-1, color='k', ls=':', label="NH3 pq")
        # plt.legend()
        # plt.savefig(output_dir+"debug_abund.png")
        # plt.close()

        mmw = np.mean(mean_molar_masses)

        if "kappa_base" in params.keys():

            cloud_scat_opas = cloud_scattering_opa(10**params["kappa_base"],10**params["p_base"], params["f_sed"], params["xi"], params["omega"])
            cloud_abs_opas = cloud_absorption_opa(10**params["kappa_base"],10**params["p_base"], params["f_sed"], params["xi"], params["omega"])
        else:
            cloud_scat_opas = None
            cloud_abs_opas = None

        if "sigma_ln" in params.keys():

            if 'f_c' in params.keys():
                cloud_fraction = params['f_c']
                complete_coverage_clouds = ['Fe(s)_crystalline__DHS'] # None
            else:
                cloud_fraction = 1.0
                complete_coverage_clouds = None
    
            cloud_particle_radius_distribution_std = params['sigma_ln']
    
            cbases = {}
            cloud_f_sed = {}
            cloud_particles_mean_radii = {}
            for specie in atmosphere.cloud_species:
    
                cloud_particles_mean_radii[specie] = 10**params['log_cloud_radius_'+specie] * np.ones_like(pressures)
    
                if 'fsed_' + specie in params.keys():
                    cloud_f_sed[specie] = params[f'fsed_{specie}']
                elif 'fsed' in params.keys():
                    cloud_f_sed[specie] = params['fsed']
                else:
                    cloud_f_sed[specie] = 0.0 # block cloud if no fsed
                easy_chem_name = specie.split('_')[0].split('-')[0].split(".")[0]
                if 'cloud_mass_fraction' + specie in params.keys():
                    cmf = 10**params['cloud_mass_fraction' + specie]
                else:
                    cmf = return_cloud_mass_fraction(specie, feh, co_ratio)
                if 'log_P_cloud_' + specie in params.keys():
                    cbase = 10**params['log_P_cloud_' + specie]
                else:
                    cbase = simple_cdf(specie, pressures, temperature, feh, co_ratio, mmw=mmw)
                cbases[easy_chem_name] = cbase
                mass_fractions_cloud = np.zeros_like(temperature)
                mass_fractions_cloud[pressures<=cbase] = cmf * (pressures[pressures<=cbase] / cbase) ** cloud_f_sed[specie]
    
                # print(mass_fractions_cloud)
                
                if "eq_scaling_" + specie in params.keys():
                    mass_fractions_cloud *= (10 ** params['eq_scaling_' + specie]) # Scaled by a constant factor
                    
                mass_fractions[specie] = mass_fractions_cloud
                
            for species in line_species:
                easy_chem_name = species.split('_')[0].split('-')[0].split(".")[0]
                if 'FeH' in species:
                    # Magic factor for FeH opacity - off by factor of 2
                    abunds_change_rainout = copy.copy(mass_fractions[species] / 2.)
                    if 'Fe(s)' in cbases.keys():
                        index_ro = pressures < cbases['Fe(s)']  # Must have iron cloud
                        abunds_change_rainout[index_ro] = 0.
                    mass_fractions[species] = abunds_change_rainout
        else:
            cloud_particles_mean_radii = None
            cloud_particle_radius_distribution_std = None
            cloud_fraction = None
            complete_coverage_clouds = None
        
                        
        # set up resolution specific mass fraction dictionaries
        mfs = copy.copy(mass_fractions)
        for key in line_species:
            mfs[key+f'.R{nsmresl}'] = mass_fractions[key.split('_')[0].split('-')[0].split(".")[0]]
            mfs.pop(key.split('_')[0].split('-')[0].split(".")[0])

        # set up resolution and wavelength range specific R-T calcs for each spectrum/dataset
        wavelengths_spherex, flux_spherex, additional_returned_quantities = atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mfs,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            additional_absorption_opacities_function=cloud_abs_opas,
            additional_scattering_opacities_function=cloud_scat_opas,
            cloud_particles_radius_distribution="lognormal",
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
            cloud_fraction=cloud_fraction,
            complete_coverage_clouds=complete_coverage_clouds,
            return_contribution=False
        )

        wavelengths = [wavelengths_spherex*1e4]
        flux = [np.nan_to_num(flux_spherex) * r2d2 * unit_conv]

        if return_extras:
            if plot:
                wavelengths_plot, flux_plot, additional_returned_quantities = atmosphere.calculate_flux(
                    temperatures=temperature,
                    mass_fractions=mfs,
                    mean_molar_masses=mean_molar_masses,
                    reference_gravity=reference_gravity,
                    additional_absorption_opacities_function=cloud_abs_opas,
                    additional_scattering_opacities_function=cloud_scat_opas,
                    cloud_particles_radius_distribution="lognormal",
                    cloud_particles_mean_radii=cloud_particles_mean_radii,
                    cloud_particle_radius_distribution_std = cloud_particle_radius_distribution_std,
                    cloud_fraction=cloud_fraction,
                    complete_coverage_clouds=complete_coverage_clouds,
                    return_contribution=True
                )
                return wavelengths, flux, wavelengths_plot*1e4, flux_plot* r2d2 * unit_conv, pressures, temperature, mass_fractions, additional_returned_quantities['emission_contribution']
            else:
                raise ValueError("Can't return extras if plot is not true")
        else:
            return wavelengths, flux


    def likelihood(param_arr, debug=False, return_spec=False):

        ln = 0

        params = copy.deepcopy(default_params)
        for i in range(len(params)):
            params[param_names[i]] = param_arr[i]

        w_i, f_i = spectrum_generator(params)

        if "f_c" in params.keys():
            patchy_params = copy.deepcopy(params)
            # throw out SiO clouds in column 2, keep Fe cloud
            patchy_params["cloud_mass_fraction_SiO2(s)_amorphous__Mie"] = -99.0
            patchy_params["cloud_mass_fraction_MgSiO3(s)_amorphous__DHS"] = -99.0
            w_i_patch, f_i_patch = spectrum_generator(patchy_params)
            f_i[0] = (f_i[0] * params["f_c"]) + (f_i_patch[0] * (1 - params["f_c"]))

        # compute nirspec likelihood:

        if 'e_hat' in params.keys():
            e_hat = params['e_hat']
        else:
            e_hat = 1

        fe_i = fe * e_hat

        rb_f_i = spectres(w, w_i[0], f_i[0])


        if debug:
            plt.figure()
            plt.plot(w, rb_f_i)

        if 'corr_len_nirspec' in params.keys():
            # from Wang et al. 2020, species.fit.fit_model
            wavel_j, wavel_i = np.meshgrid(w, w)

            error_j, error_i = np.meshgrid(fe_i, fe_i)

            corr_len = 10.0 ** params["corr_len_nirspec"]  # (um)
            corr_amp = params["corr_amp_nirspec"]

            cov_matrix = (
                corr_amp**2
                * error_i
                * error_j
                * np.exp(-((wavel_i - wavel_j) ** 2) / (2.0 * corr_len**2))
                + (1.0 - corr_amp**2) * np.eye(w.shape[0]) * error_i**2
            )

            ln_s = (
                (f - rb_f_i)
                @ np.linalg.inv(cov_matrix)
                @ (f - rb_f_i)
            )

            ln_s += np.nansum(
                np.log(2.0 * np.pi *  fe_i**2)
            )

            ln_s *= -0.5

            ln += ln_s
        
        else:
            chi2 = np.nansum(((f - rb_f_i)/fe_i)**2)
            ln += -chi2/2 - np.nansum(np.log(2*np.pi*fe_i**2)/2)

            
        if debug:
            plt.savefig(output_dir+'temp_spec.png')

        if np.isnan(np.sum(ln)):
            return -np.inf

        if return_spec:
            return ln, rb_f_i
        else:
            return ln

    default_params = {
        'R_pl':1.0,
        'plx':82.0248,
        'logg':4.5,

        'T_bottom':5000.,
        # 'N_layers':10,
        'dPT_10':0.05,
        'dPT_9':0.05,
        'dPT_8':0.05,
        'dPT_7':0.06,
        'dPT_6':0.08,
        'dPT_5':0.16,
        'dPT_4':0.21,
        'dPT_3':0.18,
        'dPT_2':0.15,
        'dPT_1':0.15,

        'C/O':0.55,
        'Fe/H':0.0,
        'log_kzz_chem':10,

        'sigma_ln' : 1.2,
        'log_cloud_radius_Fe(s)_amorphous__DHS' : -3.0,
        'cloud_mass_fraction_Fe(s)_amorphous__DHS' : -4.0,
        'log_P_cloud_Fe(s)_amorphous__DHS' : 2.0,
        'fsed_Fe(s)_amorphous__DHS' : 2.0,

        'log_cloud_radius_MgSiO3(s)_amorphous__DHS' : -0.33,
        'cloud_mass_fraction_MgSiO3(s)_amorphous__DHS' : -3.0,
        'log_P_cloud_MgSiO3(s)_amorphous__DHS' : 1.0,
        'fsed_MgSiO3(s)_amorphous__DHS' : 0.5,

        'log_cloud_radius_SiO2(s)_amorphous__Mie' : -0.33,
        'cloud_mass_fraction_SiO2(s)_amorphous__Mie' : -3.0,
        'log_P_cloud_SiO2(s)_amorphous__Mie' : 1.0,
        'fsed_SiO2(s)_amorphous__Mie' : 0.5,

        "f_c" : 0.5,

        # 'corr_len_nirspec':-1, # log10 [-3, 0] 
        # 'corr_amp_nirspec':0.5, # [0, 1]

        # 'e_hat' : 1.0
        
    }

    priors = {
        'R_pl': lambda x : uniform_prior(x, 0.5, 2.0),
        'logg': lambda x : uniform_prior(x, 3.0, 6.0),
        'plx': lambda x : gaussian_prior(x, 82.0248, 0.3583),

        'T_bottom': lambda x : uniform_prior(x, 2500, 25000),
        'dPT_1': lambda x : uniform_prior(x, 0.05, 0.25),
        'dPT_2': lambda x : gaussian_prior(x, 0.15, 0.01),
        'dPT_3': lambda x : gaussian_prior(x, 0.18, 0.04),
        'dPT_4': lambda x : gaussian_prior(x, 0.21, 0.05),
        'dPT_5': lambda x : gaussian_prior(x, 0.16, 0.06),
        'dPT_6': lambda x : gaussian_prior(x, 0.08, 0.025),
        'dPT_7': lambda x : gaussian_prior(x, 0.06, 0.1),
        'dPT_8': lambda x : uniform_prior(x, -0.05, 0.1),
        'dPT_9': lambda x : uniform_prior(x, -0.05, 0.1),
        'dPT_10': lambda x : uniform_prior(x, -0.05, 0.1),

        'C/O': lambda x : uniform_prior(x, 0.1, 1.0),
        'Fe/H': lambda x : uniform_prior(x, -0.5, 2.0),
        'log_kzz_chem': lambda x : uniform_prior(x, 4, 14),

        # 'sigma_ln': uniform_prior((1.005, 3.0))
        'sigma_ln': lambda x : delta_prior(x, 2.0, 0.0),

        'log_cloud_radius_Fe(s)_amorphous__DHS': lambda x : uniform_prior(x, -7.0, 3.0),
        'cloud_mass_fraction_Fe(s)_amorphous__DHS': lambda x : uniform_prior(x, -8.0, -0.3),
        'log_P_cloud_Fe(s)_amorphous__DHS': lambda x : uniform_prior(x, -3.0, 3.0),
        'fsed_Fe(s)_amorphous__DHS': lambda x : uniform_prior(x, 0.1, 10),

        'log_cloud_radius_MgSiO3(s)_amorphous__DHS': lambda x : uniform_prior(x, -7.0, 3.0),
        'cloud_mass_fraction_MgSiO3(s)_amorphous__DHS': lambda x : uniform_prior(x, -8.0, -0.3),
        'log_P_cloud_MgSiO3(s)_amorphous__DHS': lambda x : uniform_prior(x, -3.0, 3.0),
        'fsed_MgSiO3(s)_amorphous__DHS': lambda x : uniform_prior(x, 0.1, 10),

        'log_cloud_radius_SiO2(s)_amorphous__Mie': lambda x : uniform_prior(x, -7.0, 3.0),
        'cloud_mass_fraction_SiO2(s)_amorphous__Mie': lambda x : uniform_prior(x, -8.0, -0.3),
        'log_P_cloud_SiO2(s)_amorphous__Mie': lambda x : uniform_prior(x, -3.0, 3.0),
        'fsed_SiO2(s)_amorphous__Mie': lambda x : uniform_prior(x, 0.1, 10),
        
        'f_c': lambda x : uniform_prior(x, 0, 1),
    }

    default_param_array = list(default_params.values())

    param_names = list(default_params.keys())
    n_params = len(param_names)

    def prior_transform(cube):

        params = cube.copy()

        for i in range(n_params):
            params[i] = priors[param_names[i]](params[i])
        
        return params


    if rank==0:

        print("testing prior transform")

        random_param_array = prior_transform(np.random.uniform(size=n_params))
        print(random_param_array)

        t_start = time.time()
        test_w, test_f = spectrum_generator(default_params)
        t_end = time.time()
        print('First spectrum Generation time: {:.1f}s'.format(t_end - t_start))
        t_start = time.time()
        ln = likelihood(default_param_array)
        t_end = time.time()
        print('Likelihood time: {:.1f}s'.format(t_end - t_start))
        t_start = time.time()
        ln, test_fr = likelihood(random_param_array, return_spec=True)
        t_end = time.time()
        print('2nd Likelihood time: {:.1f}s'.format(t_end - t_start))
        df_test_f = spectres(w, test_w[0], test_f[0])
        plt.plot(w, df_test_f, label=ln, color='red')
        plt.plot(w, test_fr, label=ln, color='blue', lw=3)
        plt.errorbar(w, f, yerr=fe, label='jwst', marker='.', color='k', ls='none')
        
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(output_dir+'test_alldata_generation.png')

    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        likelihood,
        prior_transform,
        log_dir = output_dir,
        resume=resume
    )

    # have to choose the number of steps the slice sampler should take
    # after first results, this should be increased and checked for consistency.

    nsteps = 2 * len(param_names)

    # create step sampler:
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=nsteps,
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        # adaptive_nsteps=False,
        # max_nsteps=400
    )

    # run :
    result = sampler.run(min_num_live_points=nlive, dlogz=dlogz)
    sampler.print_results()

    comm.Barrier()

    if rank == 0:
        cornerplot(result)
        plt.savefig(output_dir+f'cornerplot_{retrieval_name}.pdf', dpi=300, bbox_inches='tight')