"""Original proposal; geometry/operator glyphs informed by the source-figure study.
No trained model, attention weights or performance measurements are depicted.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
root=Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'path','svg.hashsalt':'walk-cache-v2'})
fig=plt.figure(figsize=(10,6.7),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,10),ylim=(0,6.7));ax.axis('off')
ink='#26323b';blue='#087fa5';pink='#ad427c';gray='#627078'
def text(x,y,s,size=11,color=ink,ha='left',weight='normal'):
 ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.5)
def arrow(a,b,color=gray):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.4,color=color))
def region(x,y,field=False,phase=0):
 n=6
 for i in range(n):
  for j in range(n):
   col=plt.cm.coolwarm(.18+.64*((i+phase*j)/(5+5*phase))) if field else '#f3f7f8'
   ax.add_patch(Rectangle((x+i*.22,y+j*.22),.22,.22,fc=col,ec='white' if field else '#b7cbd1',lw=.7))
 ax.add_patch(Circle((x+.63,y+.78),.20,fc='white',ec=ink,lw=1))
 ax.add_patch(Rectangle((x,y),1.32,1.32,fill=False,ec=ink,lw=1))
 if not field: ax.add_patch(Rectangle((x+.44,y+.44),.44,.44,fill=False,ec=blue,lw=2))
def matrix(x,y,col,rows=4,cols=5):
 for i in range(cols):
  for j in range(rows):
   ax.add_patch(Rectangle((x+.19*i,y+.16*j),.14,.11,fc=col,alpha=.25+.65*(i+j)/(cols+rows-2),lw=0))
text(.45,6.34,'A  Encode the persistent physical context',15,weight='bold')
text(9.55,6.34,'PROPOSED DESIGN',9,gray,ha='right')
region(.7,4.45)
text(1.36,4.13,'Geometry + operator',11,ha='center')
text(1.36,3.87,'coefficients · boundary types',9,gray,ha='center')
arrow((2.2,5.1),(3.2,5.1),blue)
# Three local response glyphs imply a family rather than a single field.
for i in range(3):
 matrix(3.48+i*.16,4.65+i*.17,blue,rows=4,cols=5)
text(4.15,4.13,'Local response tokens',11,ha='center')
text(4.15,3.87,'encode boundary influence',9,gray,ha='center')
arrow((4.9,5.1),(6.35,5.1),blue)
text(5.62,5.4,'static encoder',9,gray,ha='center')
ax.add_patch(Rectangle((6.55,4.49),2.87,1.23,fc='#f4f9fb',ec=blue,lw=1))
matrix(6.75,4.77,blue);matrix(8.12,4.77,blue)
text(7.2,5.49,'K',11,blue,ha='center',weight='bold');text(8.57,5.49,'V',11,blue,ha='center',weight='bold')
text(8,4.13,'Store once per valid context',11,ha='center')
text(8,3.87,'per layer · versioned dependencies',9,gray,ha='center')
ax.plot([.45,9.55],[3.51,3.51],color='#d5dce0',lw=1)
text(.45,3.15,'B  Query the same context as the field changes',15,weight='bold')
region(.7,1.23,True,phase=.25)
text(1.36,.89,'Current state',11,ha='center')
text(1.36,.64,'forcing · interface values',9,gray,ha='center')
arrow((2.2,1.88),(3.35,1.88),pink)
text(2.76,2.18,'dynamic Q',9,pink,ha='center')
# A matrix product glyph shows the actual attention operation.
ax.add_patch(Rectangle((3.55,1.45),2.03,.91,fc='#fbf4f8',ec=pink,lw=1))
text(4.56,1.96,'Cross-attention',12,ha='center')
text(4.56,1.64,'Q reads K, V',10,pink,ha='center')
# Route cache reuse in an open right-side lane, away from captions.
ax.plot([9.57,9.8,9.8,6.1,6.1],[5.1,5.1,2.62,2.62,2.0],color=blue,lw=1.4)
arrow((6.1,2.0),(5.65,2.0),blue)
text(7.5,2.86,'reuse K, V',10,blue,ha='center')
arrow((5.7,1.56),(7.62,1.56),pink)
text(6.64,1.22,'compose + decode',9,gray,ha='center')
region(7.85,1.03,True,phase=.8)
text(8.51,.70,'Updated field',11,ha='center')
text(.45,.19,'Illustrative fields only. Global coupling is still required; cache reuse follows input and model dependencies.',9,gray)
svg_path = root/'figures/cache-architecture-v2.svg'
fig.savefig(svg_path,facecolor='white',metadata={'Date':None})
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
fig.savefig('/tmp/cache-architecture-v2.png',dpi=160,facecolor='white')
plt.close(fig)
