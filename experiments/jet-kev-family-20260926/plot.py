import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
j=json.loads((HERE/'jet-results.json').read_text());r=json.loads((HERE/'kev-reference.json').read_text())
tasks=[('sciq','SciQ'),('qnli','QNLI'),('contrastive_authorization','Policy: authorization'),('composition_held_and_or','Rule: (A or B) and C'),('composition_held_or_not','Rule: (A and B) or not C'),('tweet_offensive','TweetEval offensive'),('paws','PAWS'),('composition_held_conditional','Rule: if A then not B else C'),('mmlu','MMLU, 4-way'),('contrastive_deadline','Policy: deadline, 3-level score'),('emotion','Emotion, 6-way')]
series=[('Jev',r['jev'],'#efa300'),('Kev-27B',r['models']['kev-27b'],'#0b2545'),('Kev-9B',r['models']['kev-9b'],'#1757ad'),('Kev-4B',r['models']['kev-4b'],'#3388ff'),('Kev-0.8B',r['models']['kev-0.8b'],'#a2c5f5'),('Jet v6.2 · 4B',j,'#d51d69')]
fig,ax=plt.subplots(figsize=(13.5,8.5));fig.subplots_adjust(left=.29,right=.91,bottom=.18,top=.72)
fig.text(.04,.95,'Jet v6.2 compared with the Kev family',fontsize=23,weight='bold',color='#18212b')
fig.text(.04,.91,'Accuracy on transfer-v4 development · 656 clean questions · same question set',fontsize=12,color='#555')
y=np.arange(len(tasks))
for k,(name,data,color) in enumerate(series):
 xs=[100*data['tasks'][t] for t,_ in tasks];offset=(k-2.5)*.085
 ax.scatter(xs,y+offset,s=65 if k==5 else 35,c=color,marker='D' if k==5 else 'o',label=f'{name}   {100*data["transfer_acc"]:.2f}%',zorder=5 if k==5 else 3)
 if k==5:
  for x,yy in zip(xs,y):ax.text(102,yy,f'{x:.1f}',va='center',fontsize=10,color=color,weight='bold')
ax.set_yticks(y,[f'{label}  (n={j["counts"][t]})' for t,label in tasks]);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks(range(0,101,20),[str(n)+'%' for n in range(0,101,20)]);ax.grid(axis='x',alpha=.18);ax.tick_params(length=0,labelsize=10);ax.set_xlabel('Accuracy',labelpad=10)
for spine in ax.spines.values():spine.set_visible(False)
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.035,.875),ncol=3,frameon=False,fontsize=11,columnspacing=2.5,labelspacing=1.2)
fig.text(.04,.09,'Kev/Jev: published chart reference scores. Jet: newly measured on the released full merged v6.2 model.',fontsize=10,color='#555')
fig.text(.04,.06,'The suite has 764 total records; the chart headline uses the 656 clean rows. Source exposure differs between models.',fontsize=10,color='#555')
fig.text(.04,.03,'Development comparison, not an independent final test or a Decision Index score. No untrained-base results included.',fontsize=10,color='#555')
fig.savefig(HERE/'jet-vs-kev-family.png',dpi=160,facecolor='white');fig.savefig(HERE/'jet-vs-kev-family.svg',facecolor='white');plt.close(fig)
lines=['# Jet v6.2 versus Kev family','', 'Frozen transfer-v4 development clean subset: 656 questions. Higher accuracy and lower Brier are better.','', '| Model | Accuracy | Brier |','|---|---:|---:|']
for name,data,_ in sorted(series,key=lambda x:x[1]['transfer_acc'],reverse=True):lines.append(f'| {name} | {100*data["transfer_acc"]:.2f}% | {data["transfer_brier"]:.3f} |')
lines+=['','| Source | Jet v6.2 | Kev-4B | Delta (pp) |','|---|---:|---:|---:|']
for t,label in tasks:
 a=j['tasks'][t]*100;b=r['models']['kev-4b']['tasks'][t]*100;lines.append(f'| {label} | {a:.2f}% | {b:.2f}% | {a-b:+.2f} |')
lines+=['',*j['protocol']['limitations'],'',f'Reference: {j["protocol"]["reference_url"]}']
(HERE/'results.md').write_text('\n'.join(lines)+'\n')
