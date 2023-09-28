from alibi.explainers import ALE
import shap
import numpy as np
from src.plots.plot_RII import get_mesh_grid

def get_explainers(data, col_names, model, Qt_1, baseQ1, baseQ2, return_model = False):
  nx, ny = (100, 100)                     # mesh grid spacing
  minQ = 0.1                              # minQ
  maxQ = 10000                            # maxQ

  # split into input and output columns
  trainX, trainy = data[:, :-1], data[:, -1]

  # fit model
  model.fit(trainX, trainy)

  # Get ALE
  ale_exp = ALE(model.predict, feature_names=col_names[:-1])
  ale_exp = ale_exp.explain(trainX)

  # Get SHAP
  shap_exp = shap.TreeExplainer(model, trainX)
  shap_exp = shap_exp.shap_values(trainX, check_additivity=False)

  # Get yhat_mesh
  
  if Qt_1 is None:
    Xbase  = np.array([[baseQ1, baseQ2]])
  else:
    Xbase  = np.array([[baseQ1, baseQ2, Qt_1]])
  xv, yv, yhat_mesh = get_mesh_grid(model, Xbase, minx = minQ, maxx = maxQ, nx = nx, ny = ny)
  
  if return_model:
    return  ale_exp, shap_exp, trainX, yhat_mesh, xv, yv, model  
  else:
    return  ale_exp, shap_exp, trainX, yhat_mesh, xv, yv 