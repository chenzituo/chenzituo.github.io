"""Exact finite-dimensional cosine-flow toy; no learned model or RKHS performance claim."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent.parent / 'figures'
SEED = 20261010
rng = np.random.default_rng(SEED)
sigma = .12
phase = np.linspace(.4, 3.6*np.pi, 96)
radius = np.linspace(.3, 2.5, len(phase))
means = np.column_stack((2.0*radius*np.cos(phase), .55*radius*np.sin(phase)))
means -= means.mean(0)
C = means.T@means/len(means) + sigma**2*np.eye(2)
G_iso = np.trace(C)/2*np.eye(2)
Gs = {'Isotropic': G_iso, 'Covariance matched': C}

def drift(u, angle, G, target):
    a, s = np.cos(angle), np.sin(angle)
    if target == 'Gaussian':
        return u @ (a*s*(G-C) @ np.linalg.inv(a*a*C+s*s*G)).T
    Q = a*a*sigma**2*np.eye(2)+s*s*G
    invQ = np.linalg.inv(Q)
    B = a*s*(G-sigma**2*np.eye(2))@invQ
    centers = a*means
    logits = u@invQ@centers.T - .5*np.einsum('ni,ij,nj->n', centers, invQ, centers)
    logits -= logits.max(1,keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(1,keepdims=True)
    mu = weights@means
    return u@B.T + mu@(-s*np.eye(2)-a*B).T

def source(white, G):
    return white@np.linalg.cholesky(G).T

def integrate(z, G, target, steps=120):
    u=z.copy()
    snapshots=[u.copy()]
    angles=np.linspace(np.pi/2,0,steps+1)
    for j,(t,t2) in enumerate(zip(angles[:-1],angles[1:])):
        dt=t2-t
        k1=drift(u,t,G,target)
        k2=drift(u+dt*k1/2,t+dt/2,G,target)
        k3=drift(u+dt*k2/2,t+dt/2,G,target)
        k4=drift(u+dt*k3,t2,G,target)
        u+=dt*(k1+2*k2+2*k3+k4)/6
        if j+1 in (steps//2,steps): snapshots.append(u.copy())
    return snapshots

# Independent finite-difference check of the mixture continuity equation:
# d_theta log(p) + div(v) + v dot grad(log(p)) = 0.
def log_density(u, angle, G):
    a, s = np.cos(angle), np.sin(angle)
    Q = a*a*sigma**2*np.eye(2)+s*s*G
    delta = u[:, None, :] - a*means[None, :, :]
    logits = -.5*np.einsum('nki,ij,nkj->nk',delta,np.linalg.inv(Q),delta)
    m = logits.max(1)
    return (m+np.log(np.exp(logits-m[:,None]).mean(1))
            -np.log(2*np.pi)-.5*np.linalg.slogdet(Q)[1])

check_rng = np.random.default_rng(731)
check_points = check_rng.normal(size=(20,2))
h = 1e-5
continuity_residual = 0.
for G in Gs.values():
    for t in [.15,.6,1.3]:
        dtime=(log_density(check_points,t+h,G)-log_density(check_points,t-h,G))/(2*h)
        grad=np.zeros_like(check_points)
        div=np.zeros(len(check_points))
        for j in range(2):
            shift=np.eye(2)[j]*h
            grad[:,j]=(log_density(check_points+shift,t,G)-log_density(check_points-shift,t,G))/(2*h)
            div+=(drift(check_points+shift,t,G,'Spiral mixture')[:,j]-drift(check_points-shift,t,G,'Spiral mixture')[:,j])/(2*h)
        residual=dtime+div+np.sum(drift(check_points,t,G,'Spiral mixture')*grad,axis=1)
        continuity_residual=max(continuity_residual,float(np.max(abs(residual))))
assert continuity_residual < 1e-5, continuity_residual

# Common random numbers across source choices. Integrate expected squared angular
# velocity in the SAME Euclidean metric and SAME angle interval for every case.
N=4096
white=rng.normal(size=(N,2))
y_spiral=means[rng.integers(len(means),size=N)]+sigma*rng.normal(size=(N,2))
y_gaussian=source(rng.normal(size=(N,2)),C)
angles=np.linspace(0,np.pi/2,129)
results={}
for target,y in [('Gaussian',y_gaussian),('Spiral mixture',y_spiral)]:
    results[target]={}
    for label,G in Gs.items():
        z=source(white,G)
        errors=[]
        for t in angles:
            u=np.cos(t)*y+np.sin(t)*z
            v=drift(u,t,G,target)
            errors.append(np.sum(v*v,axis=1))
        errors=np.asarray(errors)
        energies=np.trapezoid(errors,angles,axis=0)
        coarse=np.trapezoid(errors[::2],angles[::2],axis=0)
        results[target][label]={'integrated_squared_angular_drift':float(energies.mean()),
          'mc_standard_error':float(energies.std(ddof=1)/np.sqrt(N)),
          'quadrature_65_vs_129_absolute_difference':float(abs(coarse.mean()-energies.mean()))}
        print(target,label,results[target][label],flush=True)

# Identity check of Gaussian conditional-mean formula, without a neural estimator.
u_check=rng.normal(size=(11,2));t=.63;a=np.cos(t);s=np.sin(t)
assert np.max(abs(drift(u_check,t,C,'Gaussian'))) < 1e-12
G=G_iso
Q=a*a*C+s*s*G
expected=(a*s*(G-C)@np.linalg.inv(Q)@u_check.T).T
assert np.allclose(drift(u_check,t,G,'Gaussian'),expected)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'Isotropic':'#b96635','Covariance matched':'#237b85'}
fig,axes=plt.subplots(2,4,figsize=(12,6.0),layout='constrained')
vis_white=rng.normal(size=(700,2))
for row,(label,G) in enumerate(Gs.items()):
    snapshots=integrate(source(vis_white,G),G,'Spiral mixture')
    for col,x in enumerate(snapshots+[y_spiral[:700]]):
        ax=axes[row,col]
        ax.scatter(x[:,0],x[:,1],s=3,alpha=.45,color=colors[label] if col<3 else '#45494f',rasterized=True)
        ax.set_xlim(-6.5,6.5);ax.set_ylim(-6.5,6.5);ax.set_aspect('equal')
        ax.set_xticks([-5,0,5]);ax.set_yticks([-5,0,5])
        if row==0: ax.set_title(['Source','Halfway in noise angle','Generated endpoint','Independent target samples'][col],fontsize=10)
        if col==0:ax.set_ylabel(label+'\ncoordinate 2')
        if row==1:ax.set_xlabel('coordinate 1')
fig.suptitle('Matching covariance aligns the spread; the flow still has to form the spiral',fontsize=14)
fig.savefig(FIG/'rkhs-flow-spiral-toy.png',dpi=190)
fig.savefig(HERE/'rkhs-flow-spiral-toy.pdf')
plt.close(fig)

fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained')
x=np.arange(2);width=.33
for j,label in enumerate(Gs):
    vals=[results[t][label]['integrated_squared_angular_drift'] for t in results]
    se=[results[t][label]['mc_standard_error'] for t in results]
    bars=ax.bar(x+(j-.5)*width,vals,width,color=colors[label],label=label,
      yerr=1.96*np.asarray(se),capsize=3)
    for b,v in zip(bars,vals):ax.annotate(f'{v:.2f}',(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
ax.set_xticks(x,['Gaussian target','Spiral mixture target'])
ax.set_ylabel('Integrated squared angular drift\n(common Euclidean metric)')
ax.set_title('Exact marginal velocity, with no neural-network training')
ax.legend(frameon=False,loc='upper left')
ax.set_ylim(0,max(r['integrated_squared_angular_drift'] for t in results.values() for r in t.values())*1.4)
fig.savefig(FIG/'rkhs-flow-toy-drift.png',dpi=190)
fig.savefig(HERE/'rkhs-flow-toy-drift.pdf')
plt.close(fig)

# Bounded integration check on common sources; records numerical sensitivity only.
z=source(vis_white[:100],C)
u120=integrate(z,C,'Spiral mixture',120)[-1]
u240=integrate(z,C,'Spiral mixture',240)[-1]
report={'seed':SEED,'target':'96 equally weighted isotropic Gaussian components on a stretched 2D spiral',
 'component_std':sigma,'sample_count_for_energy':N,'angle_quadrature_points':len(angles),
 'target_covariance':C.tolist(),'isotropic_source_covariance':G_iso.tolist(),
 'source_total_variance_matched':True,'velocity':'analytic conditional expectation',
 'metric':'Euclidean in fixed physical coordinates; integrated over angle 0 to pi/2',
 'results':results,'rk4_120_vs_240_endpoint_rms':float(np.sqrt(np.mean(np.sum((u120-u240)**2,axis=1)))),
 'continuity_equation_max_absolute_residual':continuity_residual,
 'scope':'Oracle covariance and velocity; no learned model, no RKHS lift, no forecast benchmark; energy is not runtime or an optimal-transport certificate.'}
(HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print('Saved figures and results',flush=True)
