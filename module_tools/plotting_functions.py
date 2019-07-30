import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.io as sio
import pandas as pd

from numpy import inf
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D

def rcparam():
    plt.rc('axes', linewidth=4.0, labelsize=16, labelpad=8)
    plt.rc('xtick.major', size=10, pad=5)  # length for x tick and padding
    plt.rc('ytick.major', size=10, pad=5)  # length for y tick and padding
    plt.rc('lines', mew=5, lw=4) # line 'marker edge width' and thickness
    plt.rc('ytick', labelsize=12)  # ytick label size
    plt.rc('xtick', labelsize=12)  # xtick label size


def plot_2d_heatmap_nb(data, title=None, axis_labels=None, axis_tick_format='%.0f', force_matching_ticks=True, save_fig=False, fig_name='HeatMap.png', **kwargs):
    rcparam()
    font = {'fontsize': 20,
            'fontweight' : 'bold',
            'verticalalignment': 'baseline'}

    # Figure Creation
    fig, ax = plt.subplots()
    if kwargs.get('invert_color', False):
        plt.set_cmap('viridis_r')
    else:
        plt.set_cmap('viridis')

    if kwargs.get('cmap', False):
        plt.set_cmap(kwargs['cmap'])

    if 'absolute_max' in kwargs:
        cax = ax.pcolormesh(data, vmin=0., vmax=kwargs['absolute_max'])
    elif 'limits' in kwargs:
        cax = ax.pcolormesh(data, vmin=kwargs['limits'][0], vmax=kwargs['limits'][1])
    else:
        cax = ax.pcolormesh(data)
    
    if kwargs.get('show_colorbar', False):
        if kwargs.get('colorbar_tick_format', False):
            cbar = fig.colorbar(cax, format=kwargs['colorbar_tick_format'])
        else:
            if kwargs.get('hi_low_only', False):
                cbar = fig.colorbar(cax, ticks=[kwargs['limits'][0], kwargs['limits'][1]])
                cbar.ax.set_yticklabels(['High', 'Low'])
            else:
                cbar = fig.colorbar(cax)
        

    # Adding labels
    if title:
        plt.title(title, fontdict=font)
    if axis_labels:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
    if force_matching_ticks:
        xticks = ax.get_xticks()[0:-1]
        ax.set_yticks(xticks)
    if kwargs.get('square_axis', False):
       #plt.axis('equal')
       ax.set_aspect('equal', 'box')
        
    ax.xaxis.set_major_formatter(FormatStrFormatter(axis_tick_format))
    ax.yaxis.set_major_formatter(FormatStrFormatter(axis_tick_format))

    if save_fig:
        plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')
    if kwargs.get('show_fig', False):
        plt.show()
    else:
        plt.close()


def plot_2d_heatmap(data, title=None, axis_labels=None, axis_tick_format='%.0f', force_matching_ticks=True, save_fig=False, fig_name='HeatMap.png', **kwargs):
    font = {'fontsize': 20, 'fontweight' : 'bold', 'verticalalignment': 'baseline'}
    rcparam()

    # Taking log of data 
    if 'take_log' in kwargs:
        if kwargs['take_log']:
            data[data == 0.] = 1e-35
            data = -np.log(data)
            data[data == inf] = np.min(data[data != inf])
            
    if 'fill_empty' in kwargs and 'axis_max' in kwargs:
        if kwargs['fill_empty']:
            if len(data) < kwargs['axis_max']:
                data = np.pad(data, ((0,kwargs['axis_max']-len(data)),(0,kwargs['axis_max']-len(data))), 'constant', constant_values=(np.max(data)))
    
    # Figure Creation
    fig, ax = plt.subplots()
    if 'absolute_max' in kwargs:
        cax = ax.pcolormesh(data, vmax=kwargs['absolute_max'], cmap='viridis_r')
    else:
        cax = ax.pcolormesh(data, cmap='viridis_r')
    
    # Adding Colorbar
    cbar = fig.colorbar(cax)

    # Adding labels
    if title:
        plt.title(title, fontdict=font)
    if axis_labels is not None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
    if 'axis_max' in kwargs:
        ax.set_xlim([0,kwargs['axis_max']])
        ax.set_ylim([0,kwargs['axis_max']])
    if force_matching_ticks:
        xticks = ax.get_xticks()
        ax.set_yticks(xticks)
    plt.grid(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter(axis_tick_format))
    ax.yaxis.set_major_formatter(FormatStrFormatter(axis_tick_format))
    
    if save_fig:
        plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()
    #plt.show()


def plot_2d_heatmap_count_grid(data, title=None, axis_labels=None, axis_tick_format='%.0f', force_matching_ticks=True, save_fig=False, fig_name='HeatMap.png', **kwargs):
    font = {'fontsize': 20, 'fontweight' : 'bold', 'verticalalignment': 'baseline'}
    rcparam()

    # Taking log of data 
    if 'fix_zeros' in kwargs:
        if kwargs['fix_zeros']:
            data = data + 1e-10
            
    if 'fill_empty' in kwargs and 'axis_max' in kwargs:
        if kwargs['fill_empty']:
            if len(data) < kwargs['axis_max']:
                data = np.pad(data, ((0,kwargs['axis_max']-len(data)),(0,kwargs['axis_max']-len(data))), 'constant', constant_values=1e-10)
    
    # Figure Creation
    fig, ax = plt.subplots()
    if 'absolute_max' in kwargs:
        cax = ax.pcolormesh(data, norm=colors.LogNorm(vmin=.1, vmax=kwargs['absolute_max']), cmap='viridis')
    else:
        cax = ax.pcolormesh(data, cmap='viridis')

    # Adding Colorbar
    cbar = fig.colorbar(cax)

    # Adding labels
    if title:
        plt.title(title, fontdict=font)
    if axis_labels is not None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
    if 'axis_max' in kwargs:
        ax.set_xlim([0,kwargs['axis_max']])
        ax.set_ylim([0,kwargs['axis_max']])
    if force_matching_ticks:
        xticks = ax.get_xticks()
        ax.set_yticks(xticks)
    plt.grid(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter(axis_tick_format))
    ax.yaxis.set_major_formatter(FormatStrFormatter(axis_tick_format))
    
    if save_fig:
        plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()
    #plt.show()

    
def plot_2d_pca_df(data_df, color_column=None, save_fig=False, fig_name=None, discrete=False, **kwargs):
    #2D plotting the components
    rcparam()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.set_cmap('viridis')
    
    # Creating plot object
    if color_column is not None:
        if discrete:
            p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, c=data_df[color_column].values,
                          cmap=cm.get_cmap('viridis', len(np.unique(data_df[color_column].values))))
            cbar = fig.colorbar(p, ticks=np.unique(data_df[color_column].values))
            cbar.ax.set_yticklabels(np.unique(data_df[color_column].values))
            cbar.set_label(kwargs.get('cbar_label', color_column))
        else:
            p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, c=data_df[color_column].values)
            cbar = fig.colorbar(p)
            cbar.set_label(kwargs.get('cbar_label', color_column))
            
    else:
        p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    if 'fig_shape' in kwargs.keys():
        fig.set_size_inches(kwargs['figshape'])
    else:
        fig.set_size_inches((14, 8))
    
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, format='png', bbox_inches='tight')
    plt.show()


def plot_2d_pca_df_marked(data_df, marked_dict, color_column=None, save_fig=False, fig_name=None, discrete=False, **kwargs):
    #2D plotting the components
    rcparam()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.set_cmap('viridis')
    
    # Creating plot object
    if color_column is not None:
        if discrete:
            p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, c=data_df[color_column].values,
                          cmap=cm.get_cmap('viridis', len(np.unique(data_df[color_column].values))))
            cbar = fig.colorbar(p, ticks=np.unique(data_df[color_column].values))
            cbar.ax.set_yticklabels(np.unique(data_df[color_column].values))
            cbar.set_label(kwargs.get('cbar_label', color_column))
        else:
            p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, c=data_df[color_column].values)
            cbar = fig.colorbar(p)
            cbar.set_label(kwargs.get('cbar_label', color_column))
            
    else:
        p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values)
        
    for key in marked_dict.keys():
        df_slice = data_df[data_df['trialParamSetNum']==marked_dict[key][0]]
        ax.scatter(df_slice['pca-1'].values, df_slice['pca-2'].values, c='r')
        ax.annotate(key, (df_slice['pca-1'].values, df_slice['pca-2'].values), 
                    size=14, weight='bold', xytext=(7, 7), textcoords='offset pixels')
    
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    if 'fig_shape' in kwargs.keys():
        fig.set_size_inches(kwargs['figshape'])
    else:
        fig.set_size_inches((14, 8))
    
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, format='png', bbox_inches='tight')
    plt.show()


def plot_2d_pca_df_exp(data_df_1, data_df_2, color_column_1=None, color_column_2=None, save_fig=False, fig_name=None, discrete=False, **kwargs):   
    # 2D plotting the components
    '''
        Setting unify_cbar to True in the arguments creates a "dual-scaled" colorbar, left corresponds to data_df_1[color_column_1], right to data_df_2[color_column_2]
        Setting this to false or leaving it undeclared (provided both color columns are defined) defaults to setting the colorbar to the overall max of the color values
    '''
    rcparam()
    if (color_column_1 is not None) or (color_column_2 is not None):
        fig, (ax, cax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[25,1], "wspace":0.1})   
    else:
        fig, ax = plt.subplots()
    
    # Getting data maxes for colorscales
    if color_column_1 is not None:
        data_1_max = np.nanmax(data_df_1[color_column_1].values)
    if color_column_2 is not None:
        data_2_max = np.nanmax(data_df_2[color_column_2].values)
    if (color_column_1 is not None) and (color_column_2 is not None):
        all_data_max = np.max([data_1_max, data_2_max])

    # Plotting simdat
    if color_column_1 is not None:
        if discrete:
            p = ax.scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c=data_df_1[color_column_1].values,
                          cmap=cm.get_cmap('viridis', len(np.unique(data_df_1[color_column_1].values))))
            cbar = plt.colorbar(p, cax=cax, ticks=np.unique(data_df_1[color_column_1].values))
            cbar.ax.set_yticklabels(np.unique(data_df_1[color_column_1].values))
        else:
            if kwargs.get('unify_cbar', False):
                p = ax.scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c=data_df_1[color_column_1].values)
                cbar = plt.colorbar(p, cax=cax)
                cbar.set_label(kwargs.get('cbar_label', color_column_1))
            else:
                p = ax.scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c=data_df_1[color_column_1].values, vmax=all_data_max)
                cbar = plt.colorbar(p, cax=cax)
                cbar.set_label(kwargs.get('cbar_label', color_column_1))
    else:
        p = ax.scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values)
    
    # Plotting expdat
    if color_column_2 is not None:
        if kwargs.get('unify_cbar', False):
            s = ax.scatter(data_df_2['pca-1'].values, data_df_2['pca-2'].values, c=data_df_2[color_column_2].values, marker='X', s=140)
            cbar.set_label("Simdat Entropy")
            cbar.ax.yaxis.set_label_position('left')
            cax2 = cax.twinx()
            cax2.set_ylim(0, np.nanmax(data_df_2[color_column_2].values))   
            cax2.set_ylabel('Expdat Entropy')
        else:
            s = ax.scatter(data_df_2['pca-1'].values, data_df_2['pca-2'].values, c=data_df_2[color_column_2].values, marker='X', s=140, vmax=all_data_max)
    else:
        s = ax.scatter(data_df_2['pca-1'].values, data_df_2['pca-2'].values, c='k', marker='X', s=140)
         
    # Labeling expdat
    if 'exp_point_label_column' in kwargs.keys():    
        point_labels = data_df_2[kwargs['exp_point_label_column']].values
        for i, txt in enumerate(point_labels):
            ax.annotate(txt, (data_df_2['pca-1'].values[i], data_df_2['pca-2'].values[i]), xytext=(7, 7), textcoords='offset pixels')
    
    # Formatting plot
    ax.set_xlabel('pca-1')
    ax.set_ylabel('pca-2')
    
    if 'fig_shape' in kwargs.keys():
        fig.set_size_inches(kwargs['figshape'])
    else:
        fig.set_size_inches((14, 8))
        
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    
    # Saving and showing plot
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, format='png', bbox_inches='tight')
    plt.show()


def plot_2d_pca_df_exp_4_subplots(data_df_1, data_df_2, color_column_1=None, color_column_2=None, save_fig=False, fig_name=None, discrete=False, **kwargs):   
    '''
        4 pane contrasting plot for PCA simdat and expdat visualization
    '''
    # Getting data maxes for colorscales
    rcparam()
    if color_column_1 is not None:
        data_1_max = np.nanmax(data_df_1[color_column_1].values)
    if color_column_2 is not None:
        data_2_max = np.nanmax(data_df_2[color_column_2].values)
    if (color_column_1 is not None) and (color_column_2 is not None):
        all_data_max = np.max([data_1_max, data_2_max])
    
    # Buildng figure
    fig = plt.figure()
    ax=[]
    cax=[]
    
    gs1 = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs1.update(left=0.02, right=0.46, top=0.94, bottom=0.50, wspace=0.07)
    ax.append(fig.add_subplot(gs1[0]))
    cax.append(fig.add_subplot(gs1[1]))
    
    gs2 = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs2.update(left=0.54, right=0.98, top=0.94, bottom=0.50, wspace=0.07)
    ax.append(fig.add_subplot(gs2[0], sharey=ax[0]))
    cax.append(fig.add_subplot(gs2[1]))
    
    gs3 = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs3.update(left=0.02, right=0.46, top=0.46, bottom=0.02, wspace=0.07)
    ax.append(fig.add_subplot(gs3[0], sharex=ax[0]))
    cax.append(fig.add_subplot(gs3[1]))
    
    gs4 = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs4.update(left=0.54, right=0.98, top=0.46, bottom=0.02, wspace=0.07)
    ax.append(fig.add_subplot(gs4[0], sharey=ax[2], sharex=ax[1]))
    cax.append(fig.add_subplot(gs4[1]))
    
    # Plotting simdat
    if color_column_1 is not None:
        p = ax[0].scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c=data_df_1[color_column_1].values, vmax=all_data_max)
        ax[0].text(.99, .99, 'simdat', horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes)
        cbar1 = plt.colorbar(p, cax=cax[0])
        cbar1.set_label(kwargs.get('cbar_label', color_column_1))
    else:
        p = ax[0].scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values)
    
    # Plotting expdat
    if color_column_2 is not None:
        # Adding simdat points for background
        #ax2.scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c=data_df_1[color_column_1].values, cmap='Greys')
        for i, gene_pair_num in enumerate(np.unique(data_df_2['gene_pair_num']), 1):
            ax[i].scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c='tab:gray')
            data_df_2_subset = data_df_2[data_df_2['gene_pair_num']==gene_pair_num]
            s = ax[i].scatter(data_df_2_subset['pca-1'].values, data_df_2_subset['pca-2'].values, c=data_df_2_subset[color_column_2].values, marker='X', s=140, vmax=all_data_max)
            cbar2 = plt.colorbar(s, cax=cax[i])
            cbar2.set_label(kwargs.get('cbar_label', color_column_2))
    else:
        for i, gene_pair_num in enumerate(np.unique(data_df_2['gene_pair_num']), 1):
            ax[i].scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, c='tab:gray')
            data_df_2_subset = data_df_2[data_df_2['gene_pair_num']==gene_pair_num]
            s = ax[i].scatter(data_df_2_subset['pca-1'].values, data_df_2_subset['pca-2'].values, c='k', marker='X', s=140)
         
    # Labeling expdat
    if 'exp_point_label_column' in kwargs.keys():
        for i, gene_pair_num in enumerate(np.unique(data_df_2['gene_pair_num']), 1):
            data_df_2_subset = data_df_2[data_df_2['gene_pair_num']==gene_pair_num]
            gene_pair = '-'.join(data_df_2_subset.iloc[0]['gene_pair'])
            ax[i].text(.99, .99, gene_pair, horizontalalignment='right', verticalalignment='top', transform=ax[i].transAxes)
            point_labels = data_df_2_subset[kwargs['exp_point_label_column']].values
            for j, txt in enumerate(point_labels):
                ax[i].annotate(txt, (data_df_2_subset['pca-1'].values[j], data_df_2_subset['pca-2'].values[j]), xytext=(7, 7), textcoords='offset pixels')
    
    # Formatting plot
    ax[0].set_ylabel('pca-2')
    ax[2].set_ylabel('pca-2')
    ax[2].set_xlabel('pca-1')
    ax[3].set_xlabel('pca-1')
    
    if 'fig_shape' in kwargs.keys():
        fig.set_size_inches(kwargs['figshape'])
    else:
        fig.set_size_inches((14, 9))
        
    if 'title' in kwargs.keys():
        fig.suptitle(kwargs['title'], size=14)
    
    # Saving and showing plot
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, format='png', bbox_inches='tight')
    plt.show()

 
def plot_3d_pca_df(data_df, color_column=None, save_fig=False, fig_name=None, discrete=False, **kwargs):
    print("Be sure to activate %matplotlib ipympl if viewing in a notebook.")
    rcparam()
    # 3D plotting the components
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Creating plot object
    if color_column is not None:
        if discrete:
            p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, data_df['pca-3'].values, 
                           c=data_df[color_column].values, cmap=cm.get_cmap('viridis', len(np.unique(data_df[color_column].values))))
            cbar = fig.colorbar(p, ticks=np.unique(data_df[color_column].values))
            cbar.ax.set_yticklabels(np.unique(data_df[color_column].values))
        else:
            p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, data_df['pca-3'].values, c=data_df[color_column].values)
            cbar = fig.colorbar(p)
            
    else:
        p = ax.scatter(data_df['pca-1'].values, data_df['pca-2'].values, data_df['pca-3'].values)
    
    ax.set_xlabel('pca-1')
    ax.set_ylabel('pca-2')
    ax.set_zlabel('pca-3')
    
    if 'fig_shape' in kwargs.keys():
        fig.set_size_inches(kwargs['figshape'])
    else:
        fig.set_size_inches((11, 6))
    
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, format='png', bbox_inches='tight')
    #plt.tight_layout()
    
    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.pause(.001)
    plt.show()


def plot_3d_pca_df_exp(data_df_1, data_df_2, color_column_1=None, color_column_2=None, save_fig=False, fig_name=None, **kwargs):
    # print("Be sure to activate %matplotlib ipympl if viewing in a notebook.")
    # 3D plotting the components
    rcparam()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    data_1_max = np.nanmax(data_df_1[color_column_1].values)
    data_2_max = np.nanmax(data_df_2[color_column_2].values)
    all_data_max = np.max([data_1_max, data_2_max])
    
    p = ax.scatter(data_df_1['pca-1'].values, data_df_1['pca-2'].values, data_df_1['pca-3'].values, c=data_df_1[color_column_1].values)
    cbar = fig.colorbar(p)
            
    # Plotting expdat
    l = ax.scatter(data_df_2['pca-1'].values, data_df_2['pca-2'].values, data_df_2['pca-3'].values, c=data_df_2[color_column_2].values, marker='X', vmax=all_data_max, s=140, depthshade=False)
    s = ax.plot(data_df_2['pca-1'].values, data_df_2['pca-2'].values, data_df_2['pca-3'].values, c='k')

    # Labeling expdat
    if 'exp_point_label_column' in kwargs.keys():    
        point_labels = data_df_2[kwargs['exp_point_label_column']].values
        for i, txt in enumerate(point_labels):
            ax.text(data_df_2['pca-1'].values[i], data_df_2['pca-2'].values[i], data_df_2['pca-3'].values[i], txt)#, xytext=(7, 7), textcoords='offset pixels')
    
    ax.set_xlabel('pca-1')
    ax.set_ylabel('pca-2')
    ax.set_zlabel('pca-3')
    
    if 'fig_shape' in kwargs.keys():
        fig.set_size_inches(kwargs['figshape'])
    else:
        fig.set_size_inches((11, 6))
    
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, format='png', bbox_inches='tight')
    plt.show()


def component_plot(df, component_column, title=None, x_tick_labels=[], x_label=None, y_label=None, save_fig=False, fig_name='TwoSubplotScatter.png'):
    rcparam()
    #plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Figure Creation
    for i, gene_pair_num in enumerate(np.unique(df['gene_pair_num'])):
        x = df[df['gene_pair_num']==gene_pair_num]['development_stage']
        y = df[df['gene_pair_num']==gene_pair_num][component_column]
        gene_pair = '-'.join(df[df['gene_pair_num']==gene_pair_num]['gene_pair'].iloc[0])
        ax.plot(x, y, '.-', label=gene_pair)
    
    plt.margins(y=0.15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    if title:
        plt.title(title, fontsize=20, weight='bold', y=1.02)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if save_fig and fig_name is not None:
        plt.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')

    plt.show()