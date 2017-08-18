import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_fname = "C:/Windows/Fonts/malgun.ttf"
#font_fname = 'C:/windows/fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
#font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
#rc('font', family='gulim')

#[f.name for f in font_manager.fontManager.ttflist]

t = np.arange(0, 12, 0.01)

title = '헤이맨'
print(title)

plt.figure
plt.plot(t)
plt.title(title)
plt.show()




