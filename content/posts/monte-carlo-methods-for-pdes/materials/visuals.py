"""Original pedagogical figures. Disk experiment is not a paper benchmark."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

root=Path(__file__).resolve().parent.parent
out=root/'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
blue='#185d83';orange='#b65d30';green='#237d69';gray='#56636e'
def save(fig,name):
 fig.savefig(out/name,dpi=180,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014',lw=1.4,ec=color,fc=color+'12'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color='#182c3a',linespacing=1.5)
def arrow(ax,start,end):
 ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.6,color=gray))

# Harmonic disk: g(theta)=cos(2theta), u(x,y)=x^2-y^2.
rng=np.random.default_rng(20261012);start=np.array([.35,.15]);eps=1e-5
pos=np.tile(start,(4096,1));active=np.ones(len(pos),bool);steps=np.zeros(len(pos),int)
paths=[start.copy()]
for iteration in range(10000):
 idx=np.flatnonzero(active)
 if not len(idx):break
 radii=1-np.linalg.norm(pos[idx],axis=1)
 done=radii<=eps;active[idx[done]]=False;idx=idx[~done];radii=radii[~done]
 theta=rng.uniform(0,2*np.pi,len(idx))
 pos[idx]+=radii[:,None]*np.column_stack([np.cos(theta),np.sin(theta)])
 steps[idx]+=1
 if 0 in idx:paths.append(pos[0].copy())
assert not active.any(), 'Walk cap reached; do not silently return truncated walks'
angles=np.arctan2(pos[:,1],pos[:,0]);samples=np.cos(2*angles);truth=float(start[0]**2-start[1]**2)
n=np.arange(1,len(samples)+1);mean=np.cumsum(samples)/n
variance=np.maximum((np.cumsum(samples**2)-n*mean**2)/np.maximum(n-1,1),0)
se=np.sqrt(variance/n)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
ax=axes[0];theta=np.linspace(0,2*np.pi,300)
ax.scatter(np.cos(theta),np.sin(theta),c=np.cos(2*theta),cmap='coolwarm',s=15,vmin=-1,vmax=1)
path=np.array(paths)
for point in path[:-1][:7]:ax.add_patch(Circle(point,1-np.linalg.norm(point),fill=False,ec=blue,alpha=.3,lw=1))
ax.plot(path[:,0],path[:,1],'-o',ms=3,lw=1.5,color=blue)
ax.scatter(*start,s=80,c=orange,zorder=5);ax.text(-1.05,1.17,'A. Jump across inscribed disks',fontsize=14)
ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.3),aspect='equal');ax.axis('off')
ax.text(0,-1.2,'Boundary value: cos(2θ) · one seeded walk',ha='center',fontsize=10)
ax=axes[1];ax.plot(n,mean,color=blue,label='Running estimate')
ax.fill_between(n,mean-2*se,mean+2*se,color=blue,alpha=.14,label='±2 estimated standard errors')
ax.axhline(truth,color=orange,ls='--',label=f'Exact harmonic value: {truth:.2f}')
ax.set(xscale='log',xlabel='Independent walks',ylabel='Estimated u(0.35, 0.15)',title='B. Average the boundary values',ylim=(-.35,.55));ax.legend(fontsize=9,loc='lower right')
save(fig,'walk-on-spheres.png')

# Poisson kernel on disk, shown as density with respect to theta.
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
theta=np.linspace(-np.pi,np.pi,1001)
for r,c in [(0,gray),(.5,blue),(.85,orange)]:
 density=(1-r*r)/(2*np.pi*(1-2*r*np.cos(theta)+r*r))
 axes[0].plot(theta,density,c=c,lw=2,label=f'query x = ({r}, 0)')
 assert abs(np.trapezoid(density,theta)-1)<1e-10
axes[0].set(xlabel='Boundary angle θ',ylabel='Exit density',title='A. One row: where does a query exit?');axes[0].legend(fontsize=10)
angles=np.linspace(-np.pi,np.pi,64,endpoint=False);radii=np.linspace(0,.85,30)
kernel=(1-radii[:,None]**2)/(1-2*radii[:,None]*np.cos(angles)[None,:]+radii[:,None]**2)
kernel/=kernel.sum(axis=1,keepdims=True)
im=axes[1].imshow(kernel,origin='lower',aspect='auto',extent=(-np.pi,np.pi,0,.85),cmap='Blues')
axes[1].set(xlabel='Boundary panel angle',ylabel='Interior query radius',title='B. Many rows: a reusable response map')
fig.colorbar(im,ax=axes[1],label='Normalized panel-weight illustration',shrink=.8)
save(fig,'response-kernel.png')

# Own conceptual comparison, not paper artwork.
fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.94,'Different local objects, different composition rules',fontsize=18,weight='bold')
ax.text(.02,.80,'WoDS',fontsize=14,color=blue,weight='bold')
box(ax,.02,.48,.26,.23,'Local exit statistics\n+ co-edge transitions')
box(ax,.37,.48,.26,.23,'Interface solve\n(I − Q) h = R g')
box(ax,.72,.48,.26,.23,'Apply local operators\nReconstruct interiors')
arrow(ax,(.28,.595),(.37,.595));arrow(ax,(.63,.595),(.72,.595))
ax.text(.02,.37,'LatentDDM',fontsize=14,color=green,weight='bold')
box(ax,.02,.04,.26,.23,'Frozen local operator\nFirst field predictions',green)
box(ax,.37,.04,.26,.23,'Learned composition\nCorrect local inputs',green)
box(ax,.72,.04,.26,.23,'Same local operator\nSecond pass + assembly',green)
arrow(ax,(.28,.155),(.37,.155));arrow(ax,(.63,.155),(.72,.155))
save(fig,'composition-paths.png')

fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.95,'Proposed architecture: cache context, update the field',fontsize=17,weight='bold')
box(ax,.02,.65,.26,.19,'Geometry / coefficients\nPDE / boundary types')
box(ax,.37,.65,.26,.19,'Static encoder\nLayer-specific K and V')
box(ax,.72,.65,.26,.19,'Versioned KV cache\nReuse while valid')
arrow(ax,(.28,.745),(.37,.745));arrow(ax,(.63,.745),(.72,.745))
box(ax,.02,.28,.26,.20,'Current state\nForcing · interface data',orange)
box(ax,.37,.28,.26,.20,'Dynamic queries\nCross-attend to context',orange)
box(ax,.72,.28,.26,.20,'Composition + decoder\nPhysical field update',orange)
arrow(ax,(.28,.38),(.37,.38));arrow(ax,(.63,.38),(.72,.38));arrow(ax,(.85,.65),(.53,.49))
ax.plot([.85,.85,.15],[.265,.18,.18],color=orange,lw=1.5)
ax.annotate('',xy=(.15,.265),xytext=(.15,.18),arrowprops=dict(arrowstyle='->',color=orange,lw=1.5))
ax.text(.5,.08,'Dynamic coupling still needs local and global communication.',ha='center',fontsize=12)
ax.text(.5,.015,'Proposed design · no cached-model timing or accuracy result yet',ha='center',fontsize=10,color=gray)
save(fig,'cache-architecture.png')

fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
for ax,mode in zip(axes,['local','global']):
 ax.set(xlim=(-.1,3.1),ylim=(-.8,3.45),aspect='equal');ax.axis('off')
 for i in range(3):
  for j in range(3):
   changed=(i==1 and j==1);affected=changed if mode=='local' else True
   color=orange if affected else blue
   ax.add_patch(FancyBboxPatch((i+.06,j+.06),.88,.88,boxstyle='round,pad=.01',ec=color,fc=color+'22',lw=2 if changed else 1))
   ax.text(i+.5,j+.5,'edit' if changed else ('recompute' if affected else 'reuse'),ha='center',va='center',fontsize=10)
 ax.set_title('Local static encoders' if mode=='local' else 'Globally mixed static encoder',fontsize=14)
 ax.text(1.5,-.32,'Only edited context K/V shown as invalid' if mode=='local' else 'One edit can invalidate every context token',ha='center',fontsize=10)
 ax.text(1.5,-.64,'Coupling / global solve must still be updated' if mode=='local' else 'Cache scope follows the dependency graph',ha='center',fontsize=10,color=gray)
save(fig,'cache-invalidation.png')
result={'seed':20261012,'walks':len(samples),'epsilon_shell':eps,'step_cap':10000,'cap_hits':int(active.sum()),'exact_value':truth,'mc_estimate':float(samples.mean()),'mc_standard_error':float(samples.std(ddof=1)/np.sqrt(len(samples))),'mean_steps':float(steps.mean()),'scope':'original harmonic-disk illustration; finite epsilon-shell; not a benchmark reproduction'}
(root/'materials/walk-example-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
