def subplot(plt, Y_X, sz_y_sz_x=(10, 10)):
  Y,X = Y_X
  sz_y, sz_x = sz_y_sz_x
  fig, axes = plt.subplots(Y, X, figsize=(X*sz_x, Y*sz_y))
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes
