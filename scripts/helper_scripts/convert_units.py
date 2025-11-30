import numpy as np
import astropy.units as u

# EoR-Spec Channel ID list
chnl_ids = [333, 337, 340, 343, 
            347, 350, 354, 357, 
            361, 365, 368]

# EoR-Spec Frequencies in GHz
eorspec_chnls = np.array([332.424, 336.061, 339.697, 343.333, 
                          346.97 , 350.606, 354.242, 357.879, 
                          361.515, 365.152, 368.788])

# Build look-up dictionary
keys = [f"f{c}" for c in chnl_ids]
chnl_dict = dict(zip(keys, eorspec_chnls))

def get_freq(channel) -> u.Quantity:
    """Return frequency for channel ID (int or 'fXXX'). Raises error if invalid."""
    # Accept integer or string
    if isinstance(channel, int):
        channel = f"f{channel}"
    if channel not in chnl_dict:
        raise RuntimeError(
            f"Channel {channel}  is not in list f{chnl_ids[0]}–f{chnl_ids[-1]}"
        )
    return chnl_dict[channel] * u.GHz

def convert_K_jysr(Intensity_K, frequency_GHz):
    '''
    Converts K to Jy/sr
    '''
    #---------------------------#
    # To convert μK to Jy/sr    #
    #---------------------------#
    # Ref: https://arxiv.org/pdf/astro-ph/0302223
    # Eq: 17 conversion factor from Jy sr−1 to µK
    # g(ν) = (24.76 Jy μK^−1 sr^−1 )^−1 [(sinh x/2)/x^2 ]^2
    # where x = ~  ν/(56.78 GHz)

    Intensity_uK = Intensity_K * 1e6
    
    # Define the constants
    conversion_factor = 24.76
    
    # Calculate x for each frequency
    x_values = frequency_GHz / 56.78
    
    # Calculate the hyperbolic sine term
    sinh_term = (np.sinh(x_values / 2) / x_values**2)**2
    
    # Calculate g(nu)
    g_nu = (1 / conversion_factor) * sinh_term # μK /(Jy sr^-1)
    
    # Divide by g_nu to go from μK to Jy/sr
    Intensity_Jy_sr = Intensity_uK/(g_nu) 
    return Intensity_Jy_sr