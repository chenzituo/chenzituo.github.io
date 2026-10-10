"""Two-mode RKHS lift: exact observations, minimum norm, and invisible directions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent.parent / 'figures'
x = np.linspace(0, 1, 1001)

def basis(q):
    q = np.asarray(q)
    return np.column_stack((np.sin(2*np.pi*q), np.sin(6*np.pi*q)))

def lift(points, values, rho):
    C = np.diag([1., rho])
    O = basis(points)
    G = O@C@O.T
    c = np.linalg.solve(G, values)
    coeff = C@O.T@c
    return coeff, c, G

def norm2(coeff, rho):
    return coeff[0]**2 + coeff[1]**2/rho

points = np.array([.25]); values = np.array([1.])
configs = [(.1, '#237b85', 'High mode penalized ($\\rho=0.1$)'),
           (1., '#b96635', 'Equal mode weights ($\\rho=1$)')]
phi = basis(x)
reference = phi[:,0]
initial = .3*phi[:,0]
eta = .5
results = {}
for rho, color, label in configs:
    coeff,c,G = lift(points,values,rho)
    expected = np.array([1,-rho])/(1+rho)
    assert np.allclose(coeff,expected,atol=1e-13)
    assert np.allclose(basis(points)@coeff,values,atol=1e-13)
    assert np.isclose(norm2(coeff,rho),values@np.linalg.solve(G,values),atol=1e-13)
    # h=phi_1+phi_3 is invisible at x=1/4 and orthogonal to the lift in H_rho.
    h = np.ones(2)
    assert np.max(abs(basis(points)@h)) < 1e-13
    assert abs(coeff@np.diag([1,1/rho])@h) < 1e-13
    assert np.isclose(norm2(coeff+.6*h,rho),norm2(coeff,rho)+.6**2*norm2(h,rho))
    # Linearity: lifting an updated observation equals updating the lifted field.
    before,_,_ = lift(points,np.array([.3]),rho)
    after,_,_ = lift(points,np.array([.3])+eta*values,rho)
    assert np.allclose(after,before+eta*coeff)
    # A second independent reading identifies both coefficients exactly.
    two = np.array([.25,1/12])
    coeff2,_,G2 = lift(two,np.array([1.,.5]),rho)
    assert np.allclose(coeff2,[1,0],atol=1e-13)
    delta=coeff-np.array([1.,0.])
    results[str(rho)]={'lift_coefficients':coeff.tolist(), 'kernel_coefficients':c.tolist(),
       'observation_gram':G.tolist(),'rkhs_norm_squared':float(norm2(coeff,rho)),
       'native_observation_max_error':float(np.max(abs(basis(points)@coeff-values))),
       'L2_error_to_declared_reference':float(np.linalg.norm(delta)/np.sqrt(2)),
       'derivative_L2_error_to_declared_reference':float(np.linalg.norm(delta*np.array([2*np.pi,6*np.pi]))/np.sqrt(2)),
       'two_sensor_coefficients':coeff2.tolist(),
       'two_sensor_gram_condition_number':float(np.linalg.cond(G2))}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
for ax in axs.flat:
    ax.axhline(0,color='#dddddd',lw=.8,zorder=0)
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Position $x$')

ax=axs[0,0]
ax.plot(x,reference,'--',color='#757575',lw=1.5,label='Reference: $\\sin(2\\pi x)$')
for rho,color,label in configs:
    coeff,_,_=lift(points,values,rho)
    ax.plot(x,phi@coeff,color=color,lw=2,label=label)
ax.scatter(points,values,color='#171c22',s=48,zorder=5,label='One observed velocity')
ax.set_title('A. Same reading, different lifted velocities',loc='left',fontweight='bold')
ax.set_ylabel('Velocity $v(x)$');ax.legend(fontsize=8,loc='lower left',framealpha=.9)

ax=axs[0,1]
ax.plot(x,initial,'--',color='#757575',lw=1.5,label='Initial field $f$')
for rho,color,label in configs:
    coeff,_,_=lift(points,values,rho)
    ax.plot(x,initial+eta*(phi@coeff),color=color,lw=2,label='$f+0.5\\,\\widetilde v$, $\\rho='+str(rho)+'$')
ax.scatter([.25],[.8],color='#171c22',s=48,zorder=5,label='Same updated reading')
ax.set_title('B. The lift specifies a whole-field update',loc='left',fontweight='bold')
ax.set_ylabel('Updated field');ax.legend(fontsize=8,loc='lower left',framealpha=.9)

ax=axs[1,0]
rho=.1;coeff,_,_=lift(points,values,rho)
base=phi@coeff;alternative=phi@(coeff+.6*np.ones(2))
ax.plot(x,base,color='#237b85',lw=2,label='Minimum-norm lift')
ax.plot(x,alternative,color='#8562a3',lw=2,label='Lift $+0.6[\\sin(2\\pi x)+\\sin(6\\pi x)]$')
ax.scatter(points,values,color='#171c22',s=48,zorder=5)
ax.set_title('C. This extra direction is invisible to the sensor',loc='left',fontweight='bold')
ax.set_ylabel('Compatible velocity');ax.legend(fontsize=8,loc='lower left',framealpha=.9)

ax=axs[1,1]
ax.plot(x,reference,color='#757575',lw=5,alpha=.4,label='Declared reference velocity')
for rho,color,label in configs:
    coeff,_,_=lift(np.array([.25,1/12]),np.array([1,.5]),rho)
    ax.plot(x,phi@coeff,color=color,lw=2,ls='-' if rho==.1 else '--',label='Two-sensor lift, $\\rho='+str(rho)+'$')
ax.scatter([.25,1/12],[1,.5],color='#171c22',s=48,zorder=5,label='Two independent readings')
ax.set_title('D. Two readings resolve both modes',loc='left',fontweight='bold')
ax.set_ylabel('Velocity $v(x)$');ax.legend(fontsize=8,loc='lower left',framealpha=.9)
fig.suptitle('An RKHS lift turns observed velocities into a function',fontsize=16)
fig.savefig(FIG/'rkhs-lift-two-mode-toy.png',dpi=190)
plt.close(fig)
(HERE/'results.json').write_text(json.dumps({'domain':[0,1],'basis':['sin(2 pi x)','sin(6 pi x)'],
  'kernel':'phi_1(x) phi_1(q) + rho phi_3(x) phi_3(q)',
  'observations':{'one_sensor':[.25],'values':[1.],'two_sensors':[.25,1/12],'values_two':[1.,.5]},
  'reference_coefficients':[1.,0.],'results':results,
  'checks':'closed-form coefficients, exact re-observation, norm identity, null-space orthogonality, Pythagoras, update linearity, two-sensor recovery',
  'scope':'Exact finite-dimensional RKHS with direct point observations. No neural training, source sampling, or performance claim.'},indent=2)+'\n')
print(json.dumps(results,indent=2))
