import numpy as np
from petitRADTRANS import physical_constants as cst

def kzz_to_co_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem_array, log10_metallicities):
    # Pressure scale height (m)
    h_scale = cst.kB * temperature / (mean_molar_masses * cst.amu * reference_gravity)

    # metallicity
    metal = 10.0**log10_metallicities

    # Diffusion coefficient (m2 s-1)
    chem_kzz = 10.0**log_kzz_chem_array

    # Mixing timescale (s)
    t_mix = h_scale**2 / chem_kzz

    # chemical timescales eq. 12-14 from Zahnle & Marley 2014
    # t_chem_co = 1.5e-6 * pressures**-1.0 * metal**-0.7 * np.exp(42000.0 / temperature)
    t_chem_1 = 1.5e-6 * pressures**-1.0 * metal**-0.7 * np.exp(42000.0 / temperature)
    t_chem_2 = 40 * pressures**-2.0 * np.exp(25000.0 / temperature)

    t_chem_co = ((1/t_chem_1)+(1/t_chem_2))**-1.0

    # Determine pressure at which t_mix = t_chem

    t_diff = t_mix - t_chem_co
    diff_product = t_diff[1:] * t_diff[:-1]

    # If t_mix and t_chem intersect then there
    # is 1 negative value in diff_product
    indices = diff_product < 0.0

    if np.sum(indices) == 1:
        p_quench = (pressures[1:] + pressures[:-1])[indices] / 2.0
        p_quench = p_quench[0]

    elif np.sum(indices) == 0:
        p_quench = None
    
    else:
        # print('found multiple p_quench intersections')
        # print(dict(zip(pressures, indices)))
        crossing = np.where(indices)[0]
        # print(crossing)
        p_quench = (pressures[1:] + pressures[:-1])[crossing] / 2.0
        # print(p_quench)
        p_quench = np.max(p_quench)
        # print(p_quench)
        # crash
    return p_quench

def kzz_to_nh3_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem_array):
    # Pressure scale height (m)
    h_scale = cst.kB * temperature / (mean_molar_masses * cst.amu * reference_gravity)

    # Diffusion coefficient (m2 s-1)
    chem_kzz = 10.0**log_kzz_chem_array

    # Mixing timescale (s)
    t_mix = h_scale**2 / chem_kzz

    # chemical timescale eq. 32 from Zahnle & Marley 2014
    t_chem_nh3 = 1e-7 * pressures**-1.0 * np.exp(52000.0 / temperature)

    # Determine pressure at which t_mix = t_chem
    t_diff = t_mix - t_chem_nh3
    diff_product = t_diff[1:] * t_diff[:-1]

    # If t_mix and t_chem intersect then there
    # is 1 negative value in diff_product
    indices = diff_product < 0.0

    if np.sum(indices) == 1:
        p_quench = (pressures[1:] + pressures[:-1])[indices] / 2.0
        p_quench = p_quench[0]

    elif np.sum(indices) == 0:
        p_quench = None
    
    else:
        crossing = np.where(indices)[0]
        p_quench = (pressures[1:] + pressures[:-1])[crossing] / 2.0
        p_quench = np.max(p_quench)
    return p_quench

def kzz_to_hcn_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem_array, log10_metallicities):
    # Pressure scale height (m)
    h_scale = cst.kB * temperature / (mean_molar_masses * cst.amu * reference_gravity)

    # metallicity
    metal = 10.0**log10_metallicities

    # Diffusion coefficient (m2 s-1)
    chem_kzz = 10.0**log_kzz_chem_array

    # Mixing timescale (s)
    t_mix = h_scale**2 / chem_kzz

    # chemical timescale eq. 40 from Zahnle & Marley 2014
    t_chem_hcn = 1.5e-4 * pressures**-1.0 * metal**-0.7 * np.exp(36000.0 / temperature)

    # Determine pressure at which t_mix = t_chem
    t_diff = t_mix - t_chem_hcn
    diff_product = t_diff[1:] * t_diff[:-1]

    # If t_mix and t_chem intersect then there
    # is 1 negative value in diff_product
    indices = diff_product < 0.0

    if np.sum(indices) == 1:
        p_quench = (pressures[1:] + pressures[:-1])[indices] / 2.0
        p_quench = p_quench[0]

    elif np.sum(indices) == 0:
        p_quench = None
    
    else:
        crossing = np.where(indices)[0]
        p_quench = (pressures[1:] + pressures[:-1])[crossing] / 2.0
        p_quench = np.max(p_quench)
    return p_quench

def kzz_to_co2_pquench(temperature, pressures, mean_molar_masses, reference_gravity, log_kzz_chem_array, log10_metallicities):
    # Pressure scale height (m)
    h_scale = cst.kB * temperature / (mean_molar_masses * cst.amu * reference_gravity)

    # Diffusion coefficient (m2 s-1)
    chem_kzz = 10.0**log_kzz_chem_array

    # Mixing timescale (s)
    t_mix = h_scale**2 / chem_kzz

    # chemical timescales eq. 12-14 from Zahnle & Marley 2014
    metal = 10.0**log10_metallicities
    t_chem_co2 = 1e-10 * pressures**-0.5 * np.exp(38000.0 / temperature)

    # Determine pressure at which t_mix = t_chem

    t_diff = t_mix - t_chem_co2
    diff_product = t_diff[1:] * t_diff[:-1]

    # If t_mix and t_chem intersect then there
    # is 1 negative value in diff_product
    indices = diff_product < 0.0

    if np.sum(indices) == 1:
        p_quench = (pressures[1:] + pressures[:-1])[indices] / 2.0
        p_quench = p_quench[0]

    elif np.sum(indices) == 0:
        p_quench = None
    
    else:
        crossing = np.where(indices)[0]
        p_quench = (pressures[1:] + pressures[:-1])[crossing] / 2.0
        p_quench = np.max(p_quench)

    return p_quench

def cloud_scattering_opa(
    kappa_base_1,
    p_base_1,
    f_sed_1,
    xi_1,
    omega_1,
):
    def get_opacity(wavelengths, pressures):
        """Wavelengths in microns, pressures in bar"""
        opacities = np.zeros((len(wavelengths), len(pressures)))
        lambda_0 = 1.0 # fix at 1 micron

        for i_p in range(len(pressures)):
            if pressures[i_p] < p_base_1:
                opacities[:, i_p] = (
                    kappa_base_1
                    * ((wavelengths / lambda_0) ** xi_1)
                    * ((pressures[i_p] / p_base_1) ** f_sed_1)
                ) * omega_1
            else:
                opacities[:, i_p] = 0.0

        return opacities

    return get_opacity

def cloud_absorption_opa(
    kappa_base_1,
    p_base_1,
    f_sed_1,
    xi_1,
    omega_1,
):
    def get_opacity(wavelengths, pressures):
        """Wavelengths in microns, pressures in bar"""
        opacities = np.zeros((len(wavelengths), len(pressures)))
        lambda_0 = 1.0 # fix at 1 micron

        for i_p in range(len(pressures)):
            if pressures[i_p] < p_base_1:
                opacities[:, i_p] = (
                    kappa_base_1
                    * ((wavelengths / lambda_0) ** xi_1)
                    * ((pressures[i_p] / p_base_1) ** f_sed_1)
                ) * (1 - omega_1)
            else:
                opacities[:, i_p] = 0.0
                
        return opacities

    return get_opacity

