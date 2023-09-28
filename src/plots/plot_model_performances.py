from settings import param_file
from matplotlib import pyplot as plt

def plot_model_performances(lag_0_list, lag_1_list, names_list, legend_labels):
  nrows           = param_file.params["fig_performance"]["nrows"] 
  ncols           = param_file.params["fig_performance"]["ncols"]  
  label_size      = param_file.params["fig_performance"]["label_size"] 
  ticklabel_size  = param_file.params["fig_performance"]["ticklabel_size"] 
  colors          = param_file.params["fig_performance"]["colors"] 
  title_size      = param_file.params["fig_performance"]["title_size"] 
  hspace          = param_file.params["fig_performance"]["hspace"] 
  wspace          = param_file.params["fig_performance"]["wspace"] 
  legend_lbl_size = param_file.params["fig_performance"]["legend_lbl_size"] 

  title_mapper    = param_file.params["fig_performance"]["titles"]


  fig, axs = plt.subplots(nrows, ncols, figsize = (12,4), sharey = True)
  axs = axs.flatten()

  for i, name in enumerate(names_list): # For each WQTP
    ax = axs[i]
    perfor_Df_0_lag = lag_0_list[i]
    perfor_Df_1_lag = lag_1_list[i]
    if i == 0:
        
      nse1, = ax.plot(perfor_Df_1_lag['NSE'], '.-', color = colors[0], label = legend_labels[1])
      nse0, = ax.plot(perfor_Df_0_lag['NSE'], '.--',  color = colors[0], label = legend_labels[0])
      
      # kge1, = ax.plot(perfor_Df_1_lag['KGE'], '.-', color = colors[1], label = legend_labels[1])
      # kge0, = ax.plot(perfor_Df_0_lag['KGE'], '.--',  color = colors[1], label = legend_labels[0])

      lognse1, = ax.plot(perfor_Df_1_lag['Log_NSE'], '.-', color = colors[1], label = legend_labels[1])
      lognse0, = ax.plot(perfor_Df_0_lag['Log_NSE'], '.--',  color = colors[1], label = legend_labels[0])
    else:
      ax.plot(perfor_Df_1_lag['NSE'], '.-', color = colors[0])
      ax.plot(perfor_Df_0_lag['NSE'], '.--',  color = colors[0])
      
      ax.plot(perfor_Df_1_lag['Log_NSE'], '.-', color = colors[1])
      ax.plot(perfor_Df_0_lag['Log_NSE'], '.--',  color = colors[1])
      
    xticks = perfor_Df_1_lag.index
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize = ticklabel_size)
    ax.tick_params(axis= 'y', labelsize = ticklabel_size)
    if i == 0:
      ax.set_ylabel('NSE, Log_NSE', fontsize = label_size)
    ax.set_title(title_mapper[name], fontsize = title_size, loc = 'left')
    ax.set_xlabel('Number of SFG', fontsize = label_size)
    

  fig.legend(handles = [nse1, nse0], title = 'NSE', 
            title_fontsize = label_size, bbox_to_anchor=(.32, 0.1), 
            loc = 'center', ncol = 2, borderaxespad=0., fontsize = label_size
            )
  fig.legend(handles = [lognse1, lognse0], title = 'Log_NSE', 
            title_fontsize = label_size, bbox_to_anchor=(.7, 0.1), 
            loc = 'center', ncol = 2, borderaxespad=0., fontsize = label_size
            )

  fig.subplots_adjust(bottom=0.33, wspace = .1)
  