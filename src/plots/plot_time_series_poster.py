import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter

from src.models.performance_metrics import calc_kge, calc_nse, calc_nse_ln
from settings import param_file


ticklabel_size  = 15
annot_labl_size = 12
y_labl_size     = 17
title_size      = 20
legend_lbl_size = 15
var_name_mapper   = param_file.params["fig_var_importances"]["variable_names"]
title_mapper      = param_file.params["fig_var_importances"]["titles"]
colors            = param_file.params["fig_time_series"]["colors"] 
fig_size = (20,15)
mgd2cms = 0.043812636574074

def plot_full_time_series(precip, usgs_flow_match, gauge_list, names_list, train_list, test_list):


  fig, axs = plt.subplots(6,2, width_ratios=[7,1], figsize = fig_size)

  axs[0,1].spines[["top", "right", "bottom", "left"]].set_visible(False)

  axs[0,1].set_xticks([])
  axs[0,1].set_yticks([])

  ax_p = axs[0,0]
  ax_Q = ax_p.twinx() 
  
  # Plot precip
  ax_p.set_ylabel('Precip\n(mm $day^{-1}$)', fontsize = y_labl_size, ha = 'center')
  # ax_p.set_title('Precipitation', loc = 'left', fontsize = title_size)
  ax_p.bar(precip.index,precip.squeeze()/10, color = 'k', width = 3)
  ax_p.set_ylim([200,0])
  ax_p.set_yticks([0, 50, 100])
  ax_p.spines.left.set_bounds(0,100)
  ax_p.set_xticklabels([])
  ax_p.tick_params(labelsize = ticklabel_size)
  # Plot Discharge
  color_idx = 1
  for gage in gauge_list["CCWQTC(MGD)"]:
     
    ax_Q.plot(usgs_flow_match[gage], 
                  color = colors[color_idx], label = var_name_mapper[gage],
                  linewidth = 1
                  )
    color_idx += 1
  ax_Q.legend(fontsize = legend_lbl_size, loc = "lower left", ncol = 2)
  ax_Q.set_yscale("log")
  ax_Q.set_ylim([0.01,1000])
  ax_Q.set_yticks([0.01, 0.1, 1, 10, 100])
  ax_Q.spines.left.set_bounds(0.01,100)
  ax_Q.set_ylabel("Discharge\n($m^{3}$ $s^{-1}$)", fontsize = y_labl_size)
  ax_Q.tick_params(labelsize = ticklabel_size)

  # fig.savefig("test.png", dpi = 300)
  plt.subplots_adjust(wspace = 0.03)
  
  for i, name in enumerate(names_list):
    
    # Get axes
    ax1 = axs[i+1,0]
    ax3 = axs[i+1,1]
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
    ax1.tick_params(labelsize = ticklabel_size)
    if i != 4:
      ax1.set_xticklabels([])


    

    # scatter plot
    sc1 = ax3.scatter(trainy, yhat_train, c = colors[0], alpha = 0.6, label = 'Train', s = 5)
    sc2 = ax3.scatter(testy,  yhat_test,  c = colors[1], alpha = 1,   label = 'Test', s = 5 )
    
    # configure ax1
    # ax1.set_xticklabels([])
    train_start = train_list[i].index[0]
    train_end   = train_list[i].index[-1]
    test_start  = test_list[i].index[0]
    test_end    = test_list[i].index[-1]
    ax1.axvline(train_end, linestyle = '--', color = 'grey')
    ax1.axvline(train_start, linestyle = '--', color = 'grey')
    ax1.axvline(test_end, linestyle = '--', color = 'grey')
    ax1.annotate(title_mapper[name][3:], xy = (0.02,0.98), xycoords = "axes fraction", fontsize = title_size, ha = "left", va = "top")
    if i ==2:
      ax1.set_ylabel('Influent Rate\n($m^{3}$ $s^{-1}$)', fontsize = y_labl_size, ha = 'center')
    
    # Configuure ax3
    adjust_scatter_axes(ax3, logscale = False)
    ax3.tick_params(labelsize = ticklabel_size)
    add_GOF_metrics(ax3, trainy, testy, yhat_train, yhat_test)
  
   
    if i == 0:
      ax3.legend(ncol = 1, bbox_to_anchor = (1.05,1.03), loc = 'lower right', fontsize = legend_lbl_size,
      handletextpad = 0.02
      )
  axs[3,1].set_ylabel('Predicted', fontsize = y_labl_size)
  # Legends
  axs[1,0].legend(ncol = 2 , loc = 'upper center')
  # xlims = ax2.get_xlim()
  # ax_p.set_xlim(xlims)

  # start_year, start_month = usgs_flow_match.index[0].year, usgs_flow_match.index[0].month_name()[0:3]
  # end_year, end_month = usgs_flow_match.index[-1].year, usgs_flow_match.index[-1].month_name()[0:3]


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
  
  ax.set_xlabel('Observed',fontsize = y_labl_size)

def add_GOF_metrics(ax, trainy, testy, yhat_train, yhat_test):
    # GOF metrics
  train_NSE  = calc_nse(trainy, yhat_train)
  test_NSE   = calc_nse(testy, yhat_test)
  train_KGE  = calc_kge(trainy, yhat_train)
  test_KGE   = calc_kge(testy, yhat_test)
  train_nseln  = calc_nse_ln(trainy, yhat_train)
  test_nseln   = calc_nse_ln(testy, yhat_test)

  # Training GOF:
  ax.annotate(f"Train\nNSE: {round(train_NSE, 2)}\nNSE-ln: {round(train_nseln, 2)}",
              xy = (0.02, 0.98), xycoords = 'axes fraction', fontsize = annot_labl_size,
              va = "top"
              )

  # Test GOF:
  ax.annotate(f"Test:\nNSE: {round(test_NSE, 2)}\nNSE-ln: {round(test_nseln, 2)}",
              xy = (0.98, 0.02), xycoords = 'axes fraction', fontsize = annot_labl_size,
              va = "bottom", ha = "right"
              )
 

  
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