"""Analytic illustration, not a WoDS or learned-model benchmark.
Run with Python, NumPy and Matplotlib. Outputs relative to this file.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
r_left, r_right = 0.5 / 1.0, 0.5 / 4.0
h = (0.0 / r_left + 1.0 / r_right) / (1.0 / r_left + 1.0 / r_right)
q_left, q_right = -h / r_left, (h - 1.0) / r_right
assert abs(q_left - q_right) < 1e-12
assert abs(h - 0.8) < 1e-12
x = np.linspace(0, 1, 501)
y = np.where(x <= 0.5, h*x/0.5, h+(1-h)*(x-0.5)/0.5)
assert y[0] == 0 and y[-1] == 1
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
ax.axvspan(0, .5, color='#e5eff9', alpha=.65)
ax.axvspan(.5, 1, color='#fbecd8', alpha=.7)
ax.plot(x, y, color='#185d83', lw=3, label='Flux-compatible interface: h = 0.8')
ax.plot(x, x, '--', color='#ae5b32', lw=2, label='Value continuity only: h = 0.5')
ax.scatter([.5,.5],[.8,.5], color=['#185d83','#ae5b32'], zorder=3)
ax.axvline(.5,color='#888888',ls=':',lw=1)
ax.text(.11,.92,'k = 1',transform=ax.transAxes)
ax.text(.76,.92,'k = 4',transform=ax.transAxes)
ax.set(xlabel='Position x',ylabel='Temperature u',xlim=(0,1),ylim=(0,1.06),title='Local solutions need a compatible interface')
ax.legend(loc='lower right',fontsize=10,frameon=True)
fig.savefig(root/'figures/interface-response.png',dpi=180)
plt.close(fig)
result={'problem':'-(k u_x)_x = 0; u(0)=0; u(1)=1',
        'conductivities':[1.,4.], 'interface_x':.5,'resistances':[r_left,r_right],
        'compatible_interface':h,'rightward_fluxes':[q_left,q_right],
        'naive_interface':.5,'naive_rightward_fluxes':[-.5/r_left,-.5/r_right],
        'max_flux_mismatch':abs(q_left-q_right),
        'scope':'analytic illustration, no Monte Carlo or model training'}
(root/'materials/interface-example-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
