import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from PIL import ImageColor
from settings import param_file
import matplotlib.ticker as ticker
from matplotlib.dates import AutoDateLocator, MonthLocator
from src.models.performance_metrics import calc_nse, calc_nse_ln
mgd2cms = 0.043812636574074

from importlib import reload
import sys
reload(sys.modules["settings"])
from settings import param_file

def get_mesh_grid(model, Xbase, minx, maxx,  nx = 100, ny = 100):

  #reference prediction 
  yhat_base_median = model.predict(Xbase)

  # meshgrid of Q1 and Q2 for predictions
  x           = np.logspace(np.floor(np.log10(minx)), np.ceil(np.log10(maxx)), nx)
  y           = np.logspace(np.floor(np.log10(minx)), np.ceil(np.log10(maxx)), ny)
  xv, yv      = np.meshgrid(x, y)
  Q_array     = np.concatenate([xv.reshape(-1,1), yv.reshape(-1,1), ], axis = 1)
  if Xbase.shape[1] == 3: # if lagged inflow is included
    Q_t_1_array = np.ones(shape=(Q_array.shape[0],1))*Xbase[0,-1]
    input_array = np.concatenate([Q_array, Q_t_1_array], axis = 1)
  else:
    input_array = Q_array
  yhat        = model.predict(input_array)
  yhat_mesh   = (yhat.reshape(xv.shape) - yhat_base_median) / yhat_base_median  

  return xv,yv, yhat_mesh

def get_cmap_and_norm(levels, color_list = None ):
  
  if color_list == None:
    color_list = ['#009E73', '#F0E442', '#E69F00']
  
  rgb_list   = [np.array(ImageColor.getrgb(c))/255 for c in color_list] # convert color list
  cmap       = ListedColormap(rgb_list)                                 # instantiate cmap
  norm       = BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True) # map colors and levels
  return cmap, norm

def plot_explainers(names_list, ale_xpl_list, shap_xpl_list, col_names_list, trainX_list, xv_list, yv_list, yhat_mesh_list):
  Thres_points =  { 'MFWQTC(MGD)' : [(2,4),(4,12),(10,30)],
                    'DRGWQTC(MGD)': [(20,60),(30,60),(70,90)],
                    'CCWQTC(MGD)' : [(20,15),(30,20),(70,40)], 
                    'FFWQTP(MGD)' : [(20,30),(25,50),(40,150)],
                    'HCWQTP(MGD)' : [(40,6),(150,15),(200,40)]
          }
  var_name_mapper   = param_file.params["fig_shap_ale_RII"]["variable_names"]
  title_mapper      = param_file.params["fig_shap_ale_RII"]["titles"]

  colors          = param_file.params["fig_shap_ale_RII"]["colors"]
  color_grid      = param_file.params["fig_shap_ale_RII"]["color_grid"]


  # tick_lbl_size   = param_file.params["fig_shap_ale_RII"]["tick_lbl_size"]
  tick_lbl_size   = 7
  lbl_size        = param_file.params["fig_shap_ale_RII"]["lbl_size"]
  title_size      = param_file.params["fig_shap_ale_RII"]["title_size"]
  legend_lbl_size = param_file.params["fig_shap_ale_RII"]["legend_lbl_size"]
  y_lbl_pad       = param_file.params["fig_shap_ale_RII"]["y_lbl_pad"]
  # xlims           = (0.1, 1000)
  xlims           = (0.005,100)



  if len(names_list) == 2:
    fig = plt.figure(figsize = (7,7), constrained_layout=False)
  else:
    fig = plt.figure(figsize = (12,7), constrained_layout=False)
    plt.subplots_adjust(wspace = 0.23)
    

  sup_gs = GridSpec(2, 1, figure = fig, height_ratios = [2,1], hspace = 0.26, wspace = 0.25)

  for i, name in enumerate(names_list):
    exp         = ale_xpl_list[i]
    shap_values = shap_xpl_list[i]
    col_names   = col_names_list[i]
    yhat_mesh   = yhat_mesh_list[i]
    xv          = xv_list[i]
    yv          = yv_list[i]

    if len(names_list) == 2:
      gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=sup_gs[0], hspace = 0.05)
      gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=sup_gs[1])
    else:
      gs1 = GridSpecFromSubplotSpec(2, 5, subplot_spec=sup_gs[0], hspace = 0.05)
      gs2 = GridSpecFromSubplotSpec(1, 5, subplot_spec=sup_gs[1], wspace = 0.25)
      # Axis ticks
    # x_major = ticker.LogLocator(base = 10.0, numticks = 5)
    x_major = ticker.FixedLocator([0.001, 0.01, 0.1, 1, 10, 100])
    x_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)

  #----------------------------------------  ALE -------------------------------
    # ax = axs[0,i]
    ax = fig.add_subplot(gs1[0, i])
    ax.plot(exp.feature_values[0]*mgd2cms, exp.ale_values[0]*mgd2cms,'.-', color = colors[0], label = var_name_mapper[col_names[0]])
    ax.plot(exp.feature_values[1]*mgd2cms, exp.ale_values[1]*mgd2cms,'.-', color = colors[1], label = var_name_mapper[col_names[1]])
    if i == 0:
      ax.set_ylabel('ALE', fontsize = lbl_size, labelpad=0.2)
    
    ax.set_xscale('log')
    ax.set_title(title_mapper[name], fontsize = title_size, loc = 'left') 
    ax.set_xlim(xlims)
    ax.set_xticklabels([])
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)
    ax.legend(fontsize = legend_lbl_size, framealpha = 0.5)
    ax.xaxis.set_major_locator(x_major)
    # ax.xaxis.set_minor_locator(x_minor)
    
    
    # ----------------------------------------SHAP ------------------------------------
    X = trainX_list[i]
    ax = fig.add_subplot(gs1[1, i])
    sc = ax.scatter(X[:,0]*mgd2cms, shap_values[:,0]*mgd2cms, alpha = 0.6, s = 10, c = colors[0], label = var_name_mapper[col_names[0]])
    sc2 = ax.scatter(X[:,1]*mgd2cms, shap_values[:,1]*mgd2cms, alpha = 0.6, s = 10, c = colors[1], label = var_name_mapper[col_names[1]])
    ax.set_xlim(xlims)
    ax.set_xscale('log')
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)

    if i == 0:
      ax.set_ylabel('Shapley Value',  fontsize = lbl_size)
    if i == 2:
      # ax.annotate("",xy=(0.08,0.36), xytext = (0.81,0.36), xycoords = "figure fraction", 
        # arrowprops={"arrowstyle" : "<->", "linewidth": 0.5}, zorder = 0
        # )
      ax.set_xlabel('Discharge ($m^{3}$ $s^{-1}$)', fontsize = lbl_size,
                    bbox={"facecolor": "white", "edgecolor":"white"}, zorder = 10
                    )

    ax.legend(fontsize = legend_lbl_size, framealpha = 0.5, 
              borderpad = 0.15,handletextpad = 0.1
              )

    
    ax.xaxis.set_major_locator(x_major)
    # ax.xaxis.set_minor_locator(x_minor)
    
    minx, maxx = ax.get_xlim()
    ax.xaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))
  # ---------------------------------------color grid -------------------
    ax = fig.add_subplot(gs2[i])
    
    levels = [0, 0.1, 0.25, 0.5, 0.75]
    cmap, norm = get_cmap_and_norm(levels, color_list = color_grid)
    c = ax.pcolormesh(xv*mgd2cms, yv*mgd2cms, yhat_mesh, cmap = cmap, norm = norm)

    #scatter with sample data  
    idx = np.random.randint(X.shape[0], size=100)
    ax.plot(X[idx,0]*mgd2cms,X[idx,1]*mgd2cms, '.k', alpha = 0.15)

    #Threshold points
    thresx, thresy = zip(*Thres_points[name]) 
    thresx, thresy = np.array(thresx), np.array(thresy)
    ax.scatter(thresx*mgd2cms,thresy*mgd2cms, color = "k", s = 13, edgecolors = "white")

    if i ==4:
      c_axs = fig.add_axes([0.91, 0.13, 0.007, 0.2])
      ticks    = [0.05, 0.175, 0.375, 0.625]
      tick_lbl = [ '< 10 %', '10 - 25 %', '25 - 50 %', '> 50 %']
      plt.colorbar(c, cax = c_axs, ticks = ticks)
      c_axs.tick_params(length=0)
      c_axs.set_yticklabels(tick_lbl)
      c_axs.annotate("$\delta Q$", xy = (2.5,1.05), xycoords = "axes fraction",fontsize = lbl_size*1.25)

    ax.set_xlim(xlims)
    ax.set_ylim(xlims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(var_name_mapper[col_names[0]] , fontsize = lbl_size*0.9, labelpad = 1, ha = 'center')
    ax.set_ylabel(var_name_mapper[col_names[1]] , labelpad = y_lbl_pad, fontsize = lbl_size*0.8, ha = 'center')
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)
    ax.xaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))
    ax.yaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))

    if i==0:
      ax.annotate("Discharge ($m^{3}$ $s^{-1}$)", xy = (-0.45,0.5), 
                xycoords = "axes fraction", ha = "center", va = "center",
                fontsize = lbl_size, rotation = 90
              )
    else: 
      ax.set_yticklabels([])
    x_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.yaxis.set_minor_locator(x_minor)
    ax.xaxis.set_major_locator(x_major)
    ax.yaxis.set_major_locator(x_major)
    ax.set_aspect("equal")

  # ax.annotate("",xy=(0.08,0.035), xytext = (0.81,0.035), xycoords = "figure fraction", arrowprops={"arrowstyle" : "<->", "linewidth": 0.5})
  ax.annotate("Discharge ($m^{3}$ $s^{-1}$)", xy = (0.45,0.035), 
                xycoords = "figure fraction", ha = "center", va = "center",
                fontsize = lbl_size, bbox={"facecolor": "white", "edgecolor":"white"}
                )

  return fig
  

def plot_explainers_with_time_series(names_list, ale_xpl_list, 
                                      shap_xpl_list, col_names_list, trainX_list, 
                                      xv_list, yv_list, yhat_mesh_list, ts_list, 
                                      OOB_scores
                                      ):

  old_Thres_points =  { 'MFWQTC(MGD)' : [(2,4),(4,12),(10,30)],
                        'DRGWQTC(MGD)': [(20,60),(30,60),(70,90)],
                        'CCWQTC(MGD)' : [(20,15),(30,20),(70,40)], 
                        'FFWQTP(MGD)' : [(20,30),(25,50),(40,150)],
                        'HCWQTP(MGD)' : [(40,6),(150,15),(200,40)]
          }
  new_Thres_points =  { 'MFWQTC(MGD)' : [(2.5,7),(4,10),(4,15)],
                        'DRGWQTC(MGD)': [(10,30),(15,30),(20,50)]
  }
  
  var_name_mapper   = param_file.params["fig_shap_ale_RII"]["variable_names"]
  
  try:
    ts_title         = param_file.params["Fig_RII_2"]["ts_ylbl"]
    ale_shap_titl    = param_file.params["Fig_RII_2"]["ale_shap_titles"]
  except:
    ts_title = {    "MFWQTC(MGD)" : "A) Morris Forman", 
                        "DRGWQTC(MGD)":  "B) Derek R. Guthrie"
                    }
    ale_shap_titl = {"MFWQTC(MGD)" : "C) Morris Forman", 
                     "DRGWQTC(MGD)": "D) Derek R. Guthrie"
                    }
    
  color_grid      = param_file.params["fig_shap_ale_RII"]["color_grid"]
  colors          = param_file.params["fig_time_series"]["colors"]

  tick_lbl_size   = 0.6 * param_file.params["fig_shap_ale_RII"]["tick_lbl_size"]
  lbl_size        = 0.6 * param_file.params["fig_shap_ale_RII"]["lbl_size"]
  title_size      = 0.6 * param_file.params["fig_shap_ale_RII"]["title_size"]
  legend_lbl_size = 0.6 * param_file.params["fig_shap_ale_RII"]["legend_lbl_size"]
  y_lbl_pad       = param_file.params["fig_shap_ale_RII"]["y_lbl_pad"]
  xlims           = (0.01, 100)

  fig = plt.figure(figsize = (3,7), constrained_layout=False)

  sup_gs = GridSpec(3, 1, figure = fig, height_ratios = [2,2,1.5], hspace = 0.3)

  for i, name in enumerate(names_list):
    exp         = ale_xpl_list[i]
    shap_values = shap_xpl_list[i]
    col_names   = col_names_list[i]
    yhat_mesh   = yhat_mesh_list[i]
    xv          = xv_list[i]
    yv          = yv_list[i]

    gs_ts = GridSpecFromSubplotSpec(2, 1, subplot_spec=sup_gs[0], hspace = 0.2)
    gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=sup_gs[1], hspace = 0.05)
    gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=sup_gs[2])
    
      # Axis ticks
    x_major = ticker.LogLocator(base = 10.0, numticks = 6)
    x_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)

  #---------------------------------------- Time Series ------------------------
    ts_df = ts_list[i]
    # try:
      # ax = fig.add_subplot(gs_ts[i], sharex = ax_share)
    # except:
    ax = fig.add_subplot(gs_ts[i])
    
    ax.plot(ts_df[name]*mgd2cms, color = colors[0], label = 'Observed', linewidth = 1)
    ax.plot(ts_df["simulated"]*mgd2cms, color = colors[1], label = 'Predicted', linewidth = 1)
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[4,6,8, 10, 12]))

    ax.set_title(ts_title[name], fontsize = lbl_size, loc = "left", pad = 1)
    # ax.set_ylabel('Influent\nRate\n($m^{3}$ $s^{-1}$)', fontsize = lbl_size)

    if i == 0:
      ax.legend(ncol = 2, loc = "upper left",fontsize = legend_lbl_size, framealpha = 0.5)
      ax_share = ax
      ax.set_xticklabels([])
      ax.annotate('Influent Rate ($m^{3}$ $s^{-1}$)', 
      xy = (-0.13,-0.05), xycoords = "axes fraction", fontsize = lbl_size*1.1, ha = "center", va = "center",
      rotation = 90
       )

    ax.annotate(f"OOB-GOF: {round(OOB_scores[i],2)}", xy = (0.97,0.86), 
                            ha = "right", xycoords = "axes fraction",
                            fontsize = tick_lbl_size)
    
    #----------------------------------------  ALE -------------------------------
    # ax = axs[0,i]
    ax = fig.add_subplot(gs1[0, i])
    ax.plot(exp.feature_values[0]*mgd2cms, exp.ale_values[0]*mgd2cms,'.-', color = colors[0], label = var_name_mapper[col_names[0]])
    ax.plot(exp.feature_values[1]*mgd2cms, exp.ale_values[1]*mgd2cms,'.-', color = colors[1], label = var_name_mapper[col_names[1]])
    if i == 0:
      ax.set_ylabel('ALE', fontsize = lbl_size*1.1)
    
    ax.set_xscale('log')
    ax.set_title(ale_shap_titl[name], fontsize = title_size, loc = 'left') 
    ax.set_xlim(xlims)
    ax.set_xticklabels([])
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)
    ax.legend(fontsize = legend_lbl_size, framealpha = 0.5)
    ax.xaxis.set_major_locator(x_major)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # ax.xaxis.set_minor_locator(x_minor)
    
    # ----------------------------------------SHAP ------------------------------------
    X = trainX_list[i]
    ax = fig.add_subplot(gs1[1, i])
    sc = ax.scatter(X[:,0]*mgd2cms, shap_values[:,0]*mgd2cms, alpha = 0.6, s = 10, c = colors[0], label = var_name_mapper[col_names[0]])
    sc2 = ax.scatter(X[:,1]*mgd2cms, shap_values[:,1]*mgd2cms, alpha = 0.6, s = 10, c = colors[1], label = var_name_mapper[col_names[1]])
    ax.set_xlim(xlims)
    ax.set_xscale('log')
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)
    if i == 0:
      ax.set_ylabel('Shapley Value',  fontsize = lbl_size*1.1)
    ax.set_xlabel('Discharge ($m^{3}$ $s^{-1}$)', fontsize = lbl_size)
    ax.legend(fontsize = legend_lbl_size, framealpha = 0.5)

    
    ax.xaxis.set_major_locator(x_major)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))
    minx, maxx = ax.get_xlim()
  # ---------------------------------------color grid -------------------
    ax = fig.add_subplot(gs2[i])
    
    levels = [0, 0.1, 0.25, 0.5, 0.75]
    cmap, norm = get_cmap_and_norm(levels, color_list = color_grid)
    c = ax.pcolormesh(xv*mgd2cms, yv*mgd2cms, yhat_mesh, cmap = cmap, norm = norm)
    
    #scatter with sample data  
    # idx = np.random.randint(X.shape[0], size=50)
    # ax.plot(X[idx,0],X[idx,1], '.k', alpha = 0.1)

    #Threshold points
    thresx, thresy = zip(*new_Thres_points[name]) 
    thresx, thresy = np.array(thresx), np.array(thresy)
    ax.scatter(thresx*mgd2cms,thresy*mgd2cms, color = "k", s = 13, edgecolors = "white")
    thresx, thresy = zip(*old_Thres_points[name]) 
    thresx, thresy = np.array(thresx), np.array(thresy)
    ax.scatter(thresx*mgd2cms,thresy*mgd2cms, s = 13, edgecolors = "k", facecolors = "none", alpha = 1, linewidths = 0.5)

    if i == 1:
      c_axs = fig.add_axes([0.93, 0.12, 0.015, 0.14])
      ticks    = [0.05, 0.175, 0.375, 0.625]
      tick_lbl = [ '< 10 %', '10 - 25 %', '25 - 50 %', '> 50 %']
      plt.colorbar(c, cax = c_axs, ticks = ticks)
      c_axs.tick_params(length=0)
      c_axs.set_yticklabels(tick_lbl, fontsize = tick_lbl_size)
      c_axs.annotate("$\delta Q$", xy = (2.5,1.05), xycoords = "axes fraction",fontsize = tick_lbl_size*1.3)


    ax.set_xlim(xlims)
    ax.set_ylim(xlims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(var_name_mapper[col_names[0]] , fontsize = lbl_size, ha = 'center')
    ax.set_ylabel(var_name_mapper[col_names[1]] , labelpad = y_lbl_pad, fontsize = lbl_size, ha = 'center')
    ax.tick_params(axis='both',  labelsize=tick_lbl_size)
    ax.xaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))
    ax.yaxis.set_major_formatter(lambda x, pos: "{val:g}".format(val = x))
    if i==0:
      ax.annotate("Discharge ($m^{3}$ $s^{-1}$)", xy = (0.59,-0.15), 
                xycoords = "figure fraction", ha = "center", va = "center",
                fontsize = lbl_size*1.1, bbox={"facecolor": "white", "edgecolor":"white", "alpha":0.0}
                )
      ax.annotate("Discharge ($m^{3}$ $s^{-1}$)", xy = (-0.35,0.5), 
                xycoords = "axes fraction", ha = "center", va = "center",
                fontsize = lbl_size*1.1, rotation = 90
              )
    else:
      ax.set_yticklabels([])

    x_minor = ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.yaxis.set_minor_locator(x_minor)
    ax.xaxis.set_major_locator(x_major)
    ax.yaxis.set_major_locator(x_major)

    ax.set_aspect("equal")