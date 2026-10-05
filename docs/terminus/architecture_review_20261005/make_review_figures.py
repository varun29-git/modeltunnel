"""Render offline diagnostics; these plots do not measure trained-model quality."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'design_measurements.json').read_text())
keys=['absmean','fixed_codes_centroid','alternating_centroid_8','row128_absmean','row64_absmean','row32_absmean']
labels=['Tile absmean','Fixed-code centroid','Alternating centroid','Row 128 absmean','Row 64 absmean','Row 32 absmean']
colors=['#bd492e','#436eaa','#327756','#7d8ca3','#536eaa','#324d84']
fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),sharey=True)
for ax,metric,title in zip(axes,['relative_rms_error','effective_rms_ratio'],
                          ['Weight reconstruction error (%)','Effective RMS / dense RMS (%)']):
    values=[100*data['offline_ternary_scale'][key][metric] for key in keys]
    ax.barh(labels,values,color=colors,height=.65)
    for i,value in enumerate(values):ax.text(value+1,i,f'{value:.1f}',va='center',fontsize=10)
    ax.set_xlim(0,105 if metric=='effective_rms_ratio' else 62)
    ax.set_title(title,fontsize=12);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis()
axes[1].axvline(100,color='#555555',linestyle='--',alpha=.6)
fig.suptitle('All 104.448M dense-parent projection weights — offline only',fontsize=13)
fig.tight_layout();fig.savefig(ROOT/'quantizer_diagnostics.png',dpi=180,bbox_inches='tight');plt.close(fig)

target=data['family']['2048'];projections=target['projection_parameters'];embedding=target['embedding_parameters']
labels=['All BF16','Ternary row-128','Ternary row-64','Ternary row-32']
proj=[2*projections/1e6]+[projections/4/1e6]*3
emb=[2*embedding/1e6]*4
scales=[0]+[target['row_group_export'][str(g)]['scale_count']*2/1e6 for g in (128,64,32)]
norms=[(target['main_norm_parameters']+target['qk_norm_parameters'])*b/1e6 for b in (2,4,4,4)]
fig,ax=plt.subplots(figsize=(9.5,4))
ax.barh(labels,proj,label='Projection weights',color='#365f91')
ax.barh(labels,emb,left=proj,label='Tied embedding/readout',color='#d2973c')
left=[a+b for a,b in zip(proj,emb)]
ax.barh(labels,scales,left=left,label='Group scales',color='#408875')
left2=[a+b for a,b in zip(left,scales)]
ax.barh(labels,norms,left=left2,color='#888888',label='Norm gains')
for i,(a,b) in enumerate(zip(left2,norms)):ax.text(a+b+25,i,f'{a+b:.0f} MB',va='center',fontsize=10)
ax.invert_yaxis();ax.set_xlim(0,3400);ax.set_xlabel('Ideal weight payload, decimal MB')
ax.set_title('Proposed 1.497B model: weight storage, excluding runtime state')
ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',fontsize=9)
fig.tight_layout();fig.savefig(ROOT/'weight_storage.png',dpi=180,bbox_inches='tight');plt.close(fig)
