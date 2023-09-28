import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter

from src.models.performance_metrics import calc_kge, calc_nse, calc_nse_ln
from settings import param_file


colors          = param_file.params["fig_time_series"]["colors"]
ticklabel_size  = param_file.params["fig_time_series"]["ticklabel_size"]
annot_labl_size = param_file.params["fig_time_series"]["annot_labl_size"]
y_labl_size     = param_file.params["fig_time_series"]["y_labl_size"]
title_size      = param_file.params["fig_time_series"]["title_size"]
legend_lbl_size = param_file.params["fig_time_series"]["legend_lbl_size"]
var_name_mapper   = param_file.params["fig_var_importances"]["variable_names"]
title_mapper      = param_file.params["fig_var_importances"]["titles"]
colors            = param_file.params["fig_time_series"]["colors"] 

mgd2cms = 0.043812636574074

def get_axs(sub_gs, fig):
  gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=sub_gs, width_ratios=[4,1], wspace=0.03)

  ax1 = fig.add_subplot(gs[0, :-1])
  ax2 = fig.add_subplot(gs[1, :-1])
  ax3 = fig.add_subplot(gs[:, -1])
  return ax1, ax2, ax3

def adjust_scatter_axes(ax, logscale = False):
  if logscale:
    xmin, ymin = ax.get_xlim()[0], ax.get_ylim()[0]
    # xmin, ymin = 1, 1
    xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
    axismin, axismax = max(xmin, ymin), max(xmax, ymax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.yaxis.set_major_locator(ticker.LogLocator(base = 10.0, numticks = 3))
    # ax.xaxis.set_major_locator(ticker.LogLocator(base = 10.0, numticks = 3))
  else:
    axismin, axismax = min(ax.get_xlim()[0], ax.get_ylim()[0]) , max(ax.get_xlim()[1], ax.get_ylim()[1]) 
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
  ax.set_xlim([axismin, axismax]) 
  ax.set_ylim([axismin, axismax])

  ax.plot([axismin,axismax],[axismin,axismax], c = 'grey', linestyle = '--')
  ax.tick_params(labelleft=False, labelright=True, right = True)  

  ax.yaxis.set_label_position("right")
  ax.set_ylabel('Predicted')
  ax.set_xlabel('Observed')

def add_GOF_metrics(ax, trainy, testy, yhat_train, yhat_test):
    # GOF metrics
  train_NSE  = calc_nse(trainy, yhat_train)
  test_NSE   = calc_nse(testy, yhat_test)
  train_KGE  = calc_kge(trainy, yhat_train)
  test_KGE   = calc_kge(testy, yhat_test)
  train_nseln  = calc_nse_ln(trainy, yhat_train)
  test_nseln   = calc_nse_ln(testy, yhat_test)

  # Training GOF:
  ax.annotate('Train: ', xy = (0.05, 0.90), xycoords = 'axes fraction', fontsize = annot_labl_size*1.1)

  ax.annotate('NSE: ' + str(round(train_NSE, 2)), 
              xy = (0.05, 0.82), xycoords = 'axes fraction', fontsize = annot_labl_size
              )
  # ax.annotate('KGE: ' + str(round(train_KGE, 2)), 
  #             xy = (0.05, 0.75), xycoords = 'axes fraction', fontsize = annot_labl_size
  #             )
  ax.annotate('NSE-ln: ' + str(round(train_nseln, 2)), 
              xy = (0.05, 0.75), xycoords = 'axes fraction', fontsize = annot_labl_size
              )
  # Test GOF:
  ax.annotate('Test: ', xy = (0.6, 0.20), xycoords = 'axes fraction', fontsize = annot_labl_size*1.1)
  ax.annotate('NSE: ' + str(round(test_NSE, 2)), 
              xy = (0.6, 0.12), xycoords = 'axes fraction', fontsize = annot_labl_size
              )
  # ax.annotate('KGE: ' + str(round(test_KGE, 2)), 
  #             xy = (0.6, 0.05), xycoords = 'axes fraction', fontsize = annot_labl_size
  #             )  
  ax.annotate('NSE-ln: ' + str(round(test_nseln, 2)), 
              xy = (0.5, 0.05), xycoords = 'axes fraction', fontsize = annot_labl_size
              )  
def plot_full_time_series(precip, usgs_flow_match, gauge_list, names_list, train_list, test_list):

  fig = plt.figure(figsize = (12,17), constrained_layout=False)
  gs = GridSpec(6, 1, figure = fig, 
                hspace = .28, bottom = 0.05, top = 0.95, left = 0.1, right = 0.95, 
                height_ratios=[0.5,1,1,1,1,1])

  # Plot precip time series
  gs_precip = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[4,1], wspace=0.03)
  ax_p = fig.add_subplot(gs_precip[0,0])
  ax_p.set_ylabel('Precip\n(mm $day^{-1}$)', fontsize = y_labl_size, ha = 'center')
  ax_p.set_title('A) Precipitation', loc = 'left', fontsize = title_size)
  ax_p.bar(precip.index,precip.squeeze()/10, color = 'k', width = 3)

  for i, name in enumerate(names_list):
    # Get axes
    ax1, ax2, ax3 = get_axs(gs[i+1], fig)

    # Prepare data
    trainy     = train_list[i]['y']
    testy      = test_list[i]['y']

    yhat_train = train_list[i]['yhat']
    yhat_test  = test_list[i]['yhat']

    y_obs  = pd.concat([train_list[i]['y'], test_list[i]['y'] ], axis = 0)
    y_pred = pd.concat([train_list[i]['yhat'], test_list[i]['yhat'] ], axis = 0)

    # Plot wqtp time series
    l1,  = ax1.plot(y_obs,     color = colors[0], label = 'Observed', linewidth = 1)
    l2,  = ax1.plot(y_pred, color = colors[1], label = 'Predicted', linewidth = 1)
    # Plot gages time series
    l3, = ax2.plot(usgs_flow_match[gauge_list[name][0]], 
                  color = colors[2], label = var_name_mapper[gauge_list[name][0]],
                  linewidth = 1
                  )
    l4, = ax2.plot(usgs_flow_match[gauge_list[name][1]], 
                  color = colors[3], label = var_name_mapper[gauge_list[name][1]],
                  linewidth = 1
                  )
    # scatter plot
    sc1 = ax3.scatter(trainy, yhat_train, c = colors[0], alpha = 0.6, label = 'Train', s = 5)
    sc2 = ax3.scatter(testy,  yhat_test,  c = colors[1], alpha = 1,   label = 'Test', s = 5 )
    
    # configure ax1
    ax1.set_xticklabels([])
    train_start = train_list[i].index[0]
    train_end   = train_list[i].index[-1]
    test_start  = test_list[i].index[0]
    test_end    = test_list[i].index[-1]
    ax1.axvline(train_end, linestyle = '--', color = 'grey')
    ax1.axvline(train_start, linestyle = '--', color = 'grey')
    ax1.axvline(test_end, linestyle = '--', color = 'grey')
    ax1.set_title(title_mapper[name], fontsize = title_size, loc = 'left')
    ax1.set_ylabel('Influent\nRate\n($m^{3}$ $s^{-1}$)', fontsize = y_labl_size, ha = 'center')

    # Configuure ax3
    adjust_scatter_axes(ax3, logscale = False)
    add_GOF_metrics(ax3, trainy, testy, yhat_train, yhat_test)
    
    # Configuure ax2
    ax2.set_ylabel('Discharge\n($m^{3}$ $s^{-1}$)', fontsize = y_labl_size, ha = 'center')
    # if i == 4:
    set_ax2_ylims(ax2, name)
    # if '03293510' in gauge_list[name]:
    #   ax2.set_yscale('symlog',linthresh=0.11)
    #   # ax2.set_ylim(-5000,5000)
    #   ax2.set_ylim(-10,100)
    #   # x_major = ticker.FixedLocator([-1000, -10, 0, 10, 1000 ])
    #   x_major = ticker.FixedLocator([-10, -1, 0, 1, 10, 100])
    #   x_minor = ticker.LogLocator()
    #   ax2.yaxis.set_major_locator(x_major)
    #   ax2.yaxis.set_minor_locator(x_minor)
    #   ax2.legend(ncol = 2, loc = 'lower right')
      
    # else:
    #   ax2.set_yscale('log')
    #   # ax2.set_yticks([10, 100, 1000])
    #   x_major = ticker.LogLocator(base = 10.0, numticks = 4)
    #   x_minor = ticker.LogLocator()
    #   ax2.yaxis.set_major_locator(x_major)
    #   ax2.yaxis.set_minor_locator(x_minor)
    #   ax2.legend(ncol = 2, loc = 'upper right')

    ax2.yaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))

    # Legends
    ax1.legend(ncol = 2 , loc = 'upper left')
    # ax2.legend(ncol = 2, loc = 'upper right')
    if i == 0:
      ax3.legend(ncol = 2, bbox_to_anchor = (0.5,1.1), loc = 'center', fontsize = legend_lbl_size)

  xlims = ax2.get_xlim()
  ax_p.set_xlim(xlims)

  start_year, start_month = usgs_flow_match.index[0].year, usgs_flow_match.index[0].month_name()[0:3]
  end_year, end_month = usgs_flow_match.index[-1].year, usgs_flow_match.index[-1].month_name()[0:3]
  # fig.suptitle(f"{start_month}/{start_year} - {end_month}/{end_year}", fontsize = 20)
  
def set_ax2_ylims(ax2, name):
  if name in ["MFWQTC(MGD)", "HCWQTP(MGD)"]:
    ax2.set_yscale('log')
    ax2.set_ylim(0.0008,30)
    x_major = ticker.FixedLocator([0.001, 0.01, 0.1, 1, 10])
    ax2.yaxis.set_major_locator(x_major)
    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    ax2.legend(ncol = 2, loc = 'upper right')
  elif name in ['CCWQTC(MGD)', 'FFWQTP(MGD)']:
    ax2.set_yscale('log')
    ax2.set_ylim(0.005,30)
    x_major = ticker.FixedLocator([0.01, 0.1, 1, 10])
    ax2.yaxis.set_major_locator(x_major)
    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    ax2.legend(ncol = 2, loc = 'upper right')
  elif name == "DRGWQTC(MGD)":
    ax2.set_yscale('symlog',linthresh=0.11)
    ax2.set_ylim(-10,100)
    ax2.set_yticks([-10, -1, 0, 1, 10, 100])
    x_major = ticker.FixedLocator([-10, -1, 0, 1, 10, 100])
    ax2.yaxis.set_major_locator(x_major)
    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    ax2.legend(ncol = 2, loc = 'lower right')      