"""Post-hoc (exploratory) check of the chance-level H1c result: index/alignment sanity + frame-head alternative.
Run: .venv\Scripts\python src/eval/study2_diag_h1c.py (from project root). Written after results were seen."""
import sys, glob, numpy as np
sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
from src.data.concept_targets import load_targets, EXPERT_BITS
from src.models.nesy import BINARY
T = load_targets(); recs=[str(r) for r in T["records"]]; pos={r:i for i,r in enumerate(recs)}
fr, cb = {}, {}
for fp in sorted(glob.glob('results/cv/nesy2_W60/r*_f*.npz')):
    z=np.load(fp,allow_pickle=True)
    for i,r in enumerate(z['records_te']):
        fr.setdefault(str(r),[]).append(z['x_f_prob'][i,0].astype(float)); cb.setdefault(str(r),[]).append(z['x_c_bits'][i])
j=EXPERT_BITS.index('accel')
ok=[r for r in recs if T['annotated'][pos[r]] and T['rec_quality'][pos[r]] and r in fr]
y=np.array([T['win_masked'][pos[r],j] for r in ok]); print('n',len(ok),'pos',(y==1).sum(),'neg',(y==0).sum(),'unk',(y<0).sum())
q=np.array([T['q'][pos[r]] for r in ok]).astype(bool)
fmax=np.array([np.mean(fr[r],0)[q[k]].max() if q[k].any() else 0 for k,r in enumerate(ok)])
fmean=np.array([np.mean(fr[r],0)[q[k]].mean() if q[k].any() else 0 for k,r in enumerate(ok)])
bit=np.array([np.mean(cb[r],0)[BINARY.index('accel')] for r in ok])
m=y>=0
for nm,s in (('frame_max',fmax),('frame_mean',fmean),('window_bit',bit)): print(nm, round(roc_auc_score(y[m],s[m]),3))
print('bit dist', np.percentile(bit,[5,50,95]).round(3))
# per-fold AUROC of window bit
au=[]
for fp in sorted(glob.glob('results/cv/nesy2_W60/r*_f*.npz')):
    z=np.load(fp,allow_pickle=True); rr=[str(r) for r in z['records_te']]
    k=[i for i,r in enumerate(rr) if r in set(ok)]
    yy=np.array([T['win_masked'][pos[rr[i]],j] for i in k]); ss=z['x_c_bits'][k,BINARY.index('accel')]
    if len(set(yy[yy>=0]))==2: au.append(roc_auc_score(yy[yy>=0],ss[yy>=0]))
print('per-fold window-bit AUROC mean/sd', np.mean(au).round(3), np.std(au).round(3))
# also no-accel / early index sanity
print('BINARY', BINARY); print('EXPERT_BITS', EXPERT_BITS)
