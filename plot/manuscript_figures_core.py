import matplotlib.pyplot as plt
from utils import utils_plot
import numpy as np
import os 
import subprocess

def plot_superresolution_grand_averages(superresolution_traces,plot_parameters):
    fig_grand_average_traces_pyr = plt.figure()
    ax_grand_average = fig_grand_average_traces_pyr.add_subplot(1,1,1)
    for sensor in plot_parameters['sensors']:
        x = superresolution_traces[sensor]['{}_time'.format(plot_parameters['trace_to_use'])]
        y = superresolution_traces[sensor]['{}_{}'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
        if plot_parameters['error_bar'] == 'sd':
            error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use'])]
        elif plot_parameters['error_bar'] == 'se':
            error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use'])]/np.sqrt(superresolution_traces[sensor]['{}_n'.format(plot_parameters['trace_to_use'])])
        else:
            error =np.zeros(len(y))
        if plot_parameters['normalize_to_max']:
            error = error/np.nanmax(y)
            y = y/np.nanmax(y)
        color = plot_parameters['sensor_colors'][sensor]
        ax_grand_average.plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = 0.9)
        if plot_parameters['error_bar'] != 'none':
            ax_grand_average.fill_between(x, y-error, y+error,color =color,alpha = .3)
    ax_grand_average.set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
    ax_grand_average.plot(0,-.05,'r|',markersize = 18)
    utils_plot.add_scalebar_to_fig(ax_grand_average,
                        axis = 'y',
                        size = plot_parameters['yscalesize'],
                        conversion = 100,
                        unit = r'dF/F%',
                        color = 'black',
                        location = 'top right',
                        linewidth = 5)
    utils_plot.add_scalebar_to_fig(ax_grand_average,
                        axis = 'x',
                        size = plot_parameters['xscalesize'],
                        conversion = 1,
                        unit = r'ms',
                        color = 'black',
                        location = 'top right',
                        linewidth = 5)
    figfile = os.path.join(plot_parameters['figures_dir'],plot_parameters['fig_name']+'.{}')
    fig_grand_average_traces_pyr.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )