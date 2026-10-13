"""Original exploratory architecture, not measured PDE outputs.
Geometric/event glyphs follow the source-figure study; no source artwork is copied.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
root=Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path','svg.hashsalt':'sequential-composition'})
fig,ax=plt.subplots(figsize=(10,6.4));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,10),ylim=(0,6.4));ax.axis('off')
blue='#087fa5';pink='#ad427c';ink='#26323b';gray='#627078'
def text(x,y,s,size=11,color=ink,ha='left',weight='normal'):
 ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.5)
def arrow(a,b,color=gray):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':color})
def domain(x,y,selected=None,field=False):
 for i in range(2):
  for j in range(2):
   n=1+i+2*j
   col=plt.cm.coolwarm(.18+.2*n) if field else ('#d6edf4' if selected==n else '#f4f6f7')
   ax.add_patch(Rectangle((x+i*.42,y+j*.42),.42,.42,fc=col,ec=blue if n==selected else '#8a989e',lw=1.4 if n==selected else .7))
   text(x+i*.42+.21,y+j*.42+.21,str(n),9,ha='center')
 ax.add_patch(Circle((x+.42,y+.42),.10,fc='white',ec=gray,lw=.7))
text(.45,6.08,'A  Condition on the full problem; append a solve history',15,weight='bold')
domain(.6,4.61)
text(1.02,4.33,'Full problem P',11,ha='center')
text(1.02,4.06,'all boundaries known',9,gray,ha='center')
for x,label,sel in [(2.5,'1: region 1',1),(4.1,'2: region 2',2),(5.7,'3: revisit 1',1)]:
 domain(x,4.61,selected=sel)
 text(x+.42,4.33,label,10,ha='center')
 if x<5.7: arrow((x+.95,5.03),(x+1.45,5.03),pink)
text(4.53,5.73,'Recorded local updates',11,ha='center')
ax.plot([2.5,2.5,6.55,6.55],[3.88,3.73,3.73,3.88],color=pink,lw=1.2)
text(4.53,3.47,'Earlier event K/V stays cached',11,pink,ha='center')
arrow((6.8,5.03),(7.62,5.03),pink)
ax.add_patch(Rectangle((7.8,4.65),1.7,.77,fc='#faf0f6',ec=pink,lw=1))
text(8.65,5.04,'Causal composer',11,ha='center')
# Full problem conditioning routes above the event glyphs.
ax.plot([1.02,1.02,8.65],[5.56,5.87,5.87],color=blue,lw=1.3)
arrow((8.65,5.87),(8.65,5.48),blue)
arrow((8.65,4.5),(8.65,2.63),pink)
text(.45,2.94,'B  Execute a local response, then record the result',15,weight='bold')
ax.add_patch(Rectangle((7.62,1.64),1.94,.78,fc='#faf0f6',ec=pink,lw=1))
text(8.59,2.12,'Next region +',11,ha='center');text(8.59,1.88,'local conditions',11,ha='center')
arrow((7.52,2.02),(6.5,2.02),pink)
ax.add_patch(Rectangle((4.54,1.64),1.85,.78,fc='#f1f7f9',ec=blue,lw=1))
text(5.465,2.12,'Local operator',11,ha='center');text(5.465,1.88,'Sθ',13,blue,ha='center')
arrow((4.4,2.02),(3.26,2.02),blue)
domain(2.28,1.6,field=True)
text(2.7,1.27,'Update current field',11,ha='center')
text(5.5,1.26,'Check interfaces / residuals',10,gray,ha='center')
# New events append after the existing prefix; no arrow points into an old event.
arrow((2.15,2.02),(1.6,2.02),pink)
ax.add_patch(Rectangle((.55,1.67),.94,.70,fc='#faf0f6',ec=pink,lw=1))
text(1.02,2.03,'event 4',10,pink,ha='center')
text(1.02,1.27,'Append',11,ha='center')
text(.45,.74,'Append event 4 after event 3, then repeat A. Earlier records and their K/V stay unchanged.',11,pink)
text(.45,.32,'Proposed solver sequence, not physical time. Fields and history are illustrative; convergence is untested.',9,gray)
p=root/'figures/sequential-composition.svg'
fig.savefig(p,metadata={'Date':None},facecolor='white')
p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
fig.savefig('/tmp/sequential-composition.png',dpi=150,facecolor='white')
plt.close(fig)
