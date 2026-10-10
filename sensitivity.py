import json,gc
import numpy as np,pandas as pd
from models import *
def ablate(x,name):
 v=x.copy();seq=v[:,:72].reshape(-1,6,12)
 if name=='no_blood_pressure':seq[:,:,3:6]=0;seq[:,:,9:12]=0
 if name=='without_explicit_masks':seq[:,:,6:12]=0
 return v
def main():
 data=raw_data();rows=[];thin=[];achieved=[];controls=[]
 obs={s:np.isfinite(x[training_indices(d)]).mean(0) for s,(d,x) in data.items()};targetobs=np.minimum(obs['mimic'],obs['eicu'])
 for source in ['mimic','eicu']:
  xx,prep,tr=prepared(source,data);sd=data[source][0];tu=np.flatnonzero(sd.split=='tune');ca=np.flatnonzero(sd.split=='calibrate')
  for name in ['no_blood_pressure','without_explicit_masks','native_death_within24h','exit_death_within24h']:
   out=A/f'sensitivity_{source}_{name}.csv'
   if out.exists():rows.extend(pd.read_csv(out).to_dict('records'));continue
   y={s:(d.y.to_numpy() if name in ['no_blood_pressure','without_explicit_masks'] else ((d.death_hour>24)&((d.death_hour if name.startswith('native') else d.los)<=48)).astype(int).to_numpy()) for s,(d,x) in data.items()}
   xp={s:ablate(x,name) for s,x in xx.items()};score=(-1,-1);records=[]
   for j,c in enumerate(CONFIGS['hist_boost']):
    est=fit('hist_boost',c,42,xp[source][tr],y[source][tr]);m=metrics(y[source][tu],predict(est,xp[source][tu],'hist_boost'));records.append(dict(candidate=j,params=json.dumps(c),**m));sc=(m['average_precision'],m['auroc'])
    if sc>score:best=est;score=sc;chosen=c
   pd.DataFrame(records).to_csv(A/f'sensitivity_tuning_{source}_{name}.csv',index=False)
   raw={s:predict(best,x,'hist_boost') for s,x in xp.items()};cal=calibrate(raw[source][ca],y[source][ca]);rr=[]
   for ev,(d,x) in data.items():
    te=np.flatnonzero(d.split=='test');rr.append(dict(train_source=source,evaluation=ev,analysis=name,model='hist_boost',params=json.dumps(chosen),**metrics(y[ev][te],apply_cal(cal,raw[ev])[te])))
   pd.DataFrame(rr).to_csv(out,index=False);rows.extend(rr);log('Sensitivity '+source+' '+name)
  # Positive-rate medication labels are not used in the redesigned outcome.
  for f in FAMILIES:
   sys=joblib.load(P/f'{source}_{f}_42_system.joblib')
   for ev,(d,x) in data.items():
    te=np.flatnonzero(d.split=='test');base=x[te];prob=np.divide(targetobs,obs[ev],out=np.ones_like(targetobs),where=obs[ev]>0);prob=np.clip(prob,0,1)
    for seed in [42,117,2026]:
     keep=np.random.default_rng(seed).random(base.shape)<prob;pert=np.where(keep,base,np.nan);xp=transform(pert,d.iloc[te],sys['prep']);p=apply_cal(sys['calibrator'],predict(sys['model'],xp,f));thin.append(dict(train_source=source,evaluation=ev,model=f,seed=seed,**metrics(d.y.to_numpy()[te],p)))
     if source=='mimic' and f==FAMILIES[0]:
      for j,feat in enumerate(['hr','rr','spo2','sbp','dbp','map']):achieved.append(dict(source=ev,feature=feat,seed=seed,native_observed=np.isfinite(base[:,:,j]).mean(),target_observed=targetobs[:,j].mean(),achieved_observed=np.isfinite(pert[:,:,j]).mean()))
   del sys;gc.collect()
  # Nonphysiological comparator; no tuning against target labels.
  y=sd.y.to_numpy();base=LogisticRegression(C=1,max_iter=500).fit(xx[source][tr,72:75],y[tr]);cal=calibrate(base.predict_proba(xx[source][ca,72:75])[:,1],y[ca])
  for ev,(d,x) in data.items():
   te=np.flatnonzero(d.split=='test');p=apply_cal(cal,base.predict_proba(xx[ev][te,72:75])[:,1]);controls.append(dict(train_source=source,evaluation=ev,analysis='age_sex_only',**metrics(d.y.to_numpy()[te],p)))
  # Deliberately shuffled source labels: evaluate true labels only after fitting.
  yp=np.random.default_rng(1942).permutation(y[tr]);est=fit('hist_boost',{'leaves':15,'lr':.03},42,xx[source][tr],yp)
  for ev,(d,x) in data.items():
   te=np.flatnonzero(d.split=='test');p=predict(est,xx[ev][te],'hist_boost');controls.append(dict(train_source=source,evaluation=ev,analysis='shuffled_training_labels',**metrics(d.y.to_numpy()[te],p)))
 pd.DataFrame(rows).to_csv(A/'sensitivities.csv',index=False);pd.DataFrame(thin).to_csv(A/'thinning.csv',index=False);pd.DataFrame(achieved).to_csv(A/'thinning_availability.csv',index=False);pd.DataFrame(controls).to_csv(A/'controls.csv',index=False)
if __name__=='__main__':main()
