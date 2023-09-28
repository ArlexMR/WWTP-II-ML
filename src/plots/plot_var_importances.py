from settings import param_file
from matplotlib import pyplot as plt


def plot_var_importances(importance_list, variable_names, wwtp_names):
  nrows           = param_file.params["fig_var_importances"]["nrows"] 
  ncols           = param_file.params["fig_var_importances"]["ncols"]  
  label_size      = param_file.params["fig_var_importances"]["label_size"] * 0.9
  ticklabel_size  = param_file.params["fig_var_importances"]["ticklabel_size"] * 0.9
  colors          = param_file.params["fig_var_importances"]["colors"] 
  title_size      = param_file.params["fig_var_importances"]["title_size"] * 0.9
  hspace          = param_file.params["fig_var_importances"]["hspace"] 
  wspace          = param_file.params["fig_var_importances"]["wspace"] 
  legend_lbl_size = param_file.params["fig_var_importances"]["legend_lbl_size"] 

  var_name_mapper   = param_file.params["fig_var_importances"]["variable_names"]
  title_mapper      = param_file.params["fig_var_importances"]["titles"]

  # msd_names   = msd_match.columns

  fig, axs = plt.subplots(nrows, ncols, figsize = (12,3), sharey = True)
  
  axs = axs.flatten()

  for i, var_importance in enumerate(importance_list):
    ax = axs[i]
    
    xticks      = [j  for j in range(len(var_importance))]
    try:
      xticklabels = [var_name_mapper[var_name] for var_name in variable_names]
    except:
      xticklabels = variable_names

    ax.bar(xticks, var_importance, color = colors[0])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize = ticklabel_size, rotation = 90)
    ax.tick_params(axis= 'y', labelsize = ticklabel_size)
    if i == 0:
      ax.set_ylabel('Var. Importance', fontsize = label_size)
    ax.set_title(title_mapper[wwtp_names[i]][3:], fontsize = title_size, loc = 'left')
    ax.set_ylim(0,0.62)

  fig.tight_layout()