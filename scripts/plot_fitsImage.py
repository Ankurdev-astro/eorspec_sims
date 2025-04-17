'''
Script to plot ML Output fits maps.
Usage: python plot_fitsImage.py -h
python plot_fitsImage.py --fits-file './file/path' --save './plot_dir/plot_1.png'
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

def plot_fits(fits_file, vmin, vmax, save):
    with fits.open(fits_file) as hdul:
        wcs = WCS(hdul[0].header, naxis=[1, 2])
        image_data = hdul[0].data.squeeze()
    image_data[image_data == 0] = np.nan

    cmap = plt.get_cmap('plasma')
    cmap.set_bad(color='gray', alpha=1.)

    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(projection=wcs)

    im = ax.imshow(image_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('Intensity [K]')

    ax.coords[0].set_format_unit('deg', decimal=True)
    ax.coords[1].set_format_unit('deg', decimal=True)
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)

    ax.set_xlabel('Right Ascension (degrees)')
    ax.set_ylabel('Declination (degrees)')

    if save:
        plt.savefig(save)
        #plt.show()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot a FITS image with WCS axes in decimal degrees.")
    parser.add_argument('fits_file', type=str, 
                        help="Path to the FITS file.")
    parser.add_argument('--vmin', type=float, 
                        default=-3e-4, help="Minimum data value for colormap.")
    parser.add_argument('--vmax', type=float, 
                        default=3e-4, help="Maximum data value for colormap.")
    parser.add_argument('--save', type=str, 
                        help="Path to save the output plot.")

    args = parser.parse_args()
    
    print(args.fits_file)
    plot_fits(args.fits_file, args.vmin, args.vmax, args.save)

if __name__ == '__main__':
    main()
