"""Recompute dependent diagnostics for the locked final models; no selection."""
from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tuning_extension'))
from run_search import TunedTransformer
import models as m
import evaluate as ev
import coverage_hospital as ch
import figures
OUT=ROOT/'integrated_analysis'; A=OUT/'results'; P=OUT/'private'
A.mkdir(parents=True,exist_ok=True);P.mkdir(exist_ok=True)
old=ROOT/'results'; new=ROOT/'tuning_extension/results'; priv=ROOT/'tuning_extension/private'
lock=json.loads((new/'locked_selection.json').read_text())['selection']
for s in ['mimic','eicu']:
 for f in m.FAMILIES:shutil.copy2(priv/f'{s}_{f}_42_predictions.npz',P/f'{s}_{f}_42.npz')
 (A/f'{s}_selected.json').write_text(json.dumps({'model':lock[s]['selected_family']}))
ev.A=A;ev.P=P;ev.main()
# Preserve the canonical target updating directly from raw probabilities.
c=pd.read_csv(new/'conformal.csv')
c['method']=c.method.replace({'marginal':'marginal_failure_baseline','target_updated_class_conditional':'local_class_conditional'})
c=c.rename(columns={f'{a}_wilson_{b}':f'{z}_{b}' for a,z in [('coverage','overall'),('event_coverage','event'),('nonevent_coverage','nonevent')] for b in ['lower','upper']})
data=m.raw_data()
for i,r in c.iterrows():
 d=data[r.evaluation if r.method=='local_class_conditional' else r.train_source][0]; y=d.loc[d.split=='conformal','y']
 c.loc[i,'calibration_n']=len(y);c.loc[i,'calibration_events']=int(y.sum())
c.to_csv(A/'conformal.csv',index=False)
shutil.copy2(new/'local_calibration.csv',A/'local_calibration.csv')
ch.A=A;ch.P=P;ch.main()
obs={s:np.isfinite(x[m.training_indices(d)]).mean(0) for s,(d,x) in data.items()};target=np.minimum(obs['mimic'],obs['eicu']);rows=[];ach=[]
for s in ['mimic','eicu']:
 prep=json.loads((priv/f'{s}_preprocessing.json').read_text());sd=data[s][0];ca=np.flatnonzero(sd.split=='calibrate');xp=m.transform(data[s][1],sd,prep)
 for f in m.FAMILIES:
  model=joblib.load(priv/f'{s}_{f}_{lock[s][f]["candidate"]}_42.joblib');cal=m.calibrate(m.predict(model,xp[ca],f),sd.y.to_numpy()[ca])
  for e,(d,x) in data.items():
   te=np.flatnonzero(d.split=='test');base=x[te];prob=np.clip(np.divide(target,obs[e],out=np.ones_like(target),where=obs[e]>0),0,1)
   for seed in [42,117,2026]:
    pert=np.where(np.random.default_rng(seed).random(base.shape)<prob,base,np.nan);p=m.apply_cal(cal,m.predict(model,m.transform(pert,d.iloc[te],prep),f))
    rows.append(dict(train_source=s,evaluation=e,model=f,seed=seed,**m.metrics(d.y.to_numpy()[te],p)))
    if s=='mimic' and f==m.FAMILIES[0]:
     for j,feat in enumerate(['hr','rr','spo2','sbp','dbp','map']):ach.append(dict(source=e,feature=feat,seed=seed,native_observed=np.isfinite(base[:,:,j]).mean(),target_observed=target[:,j].mean(),achieved_observed=np.isfinite(pert[:,:,j]).mean()))
  print('thinning',s,f,flush=True)
pd.DataFrame(rows).to_csv(A/'thinning.csv',index=False);pd.DataFrame(ach).to_csv(A/'thinning_availability.csv',index=False)
for name in ['missingness.csv','mimic_density.csv','eicu_density.csv','sensitivities.csv']:shutil.copy2(old/name,A/name)
figures.A=A;figures.F=OUT/'figures';figures.F.mkdir(exist_ok=True);figures.main()
(OUT/'README.md').write_text('Final-model diagnostics and figures. Selections are inherited unchanged from the locked development search. Target updating uses the canonical raw-score calibration. Ancillary eight-candidate task and feature ablations retain their independently specified budgets. No test results determine selection. Private files contain patient-level predictions and must not be distributed.\n')
print('INTEGRATED ANALYSIS COMPLETE',flush=True)
