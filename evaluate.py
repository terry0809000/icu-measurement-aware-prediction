from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
from scipy.special import logit,expit
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from models import *
from statistics_core import FastMetric,conformal_q,conformal_metrics,calibration_stats
B=1000
def wilson(k,n):
 if n==0:return (np.nan,np.nan)
 z=norm.ppf(.975);p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return max(0,mid-half),min(1,mid+half)
def coverage(y,p,q0,q1):
 r=conformal_metrics(y,p,q0,q1);inc=np.where(y==1,1-p<=q1,p<=q0)
 for name,ix in [('overall',np.ones(len(y),bool)),('event',y==1),('nonevent',y==0)]:
  lo,hi=wilson(int(inc[ix].sum()),int(ix.sum()));r[name+'_lower']=lo;r[name+'_upper']=hi
 return r
def bootstrap(d,preds,tag,weights=None,cluster='patient_id'):
 y=d.y.to_numpy();codes,groups=pd.factorize(d[cluster]);n=len(groups);rng=np.random.default_rng(int(h(tag+'|'+cluster,'bootstrap')*2**32));names=list(preds)
 fs=[FastMetric(y,preds[f]) for f in names];base=np.ones(len(y)) if weights is None else weights
 point=np.array([f(base) for f in fs]);draw=np.empty((B,len(names),3))
 for b in range(B):
  counts=rng.multinomial(n,np.full(n,1/n));w=base*counts[codes]
  for j,f in enumerate(fs):draw[b,j]=f(w)
 rows=[];pairs=[]
 for j,f in enumerate(names):
  for k,met in enumerate(['auroc','average_precision','brier']):
   lo,hi=np.nanquantile(draw[:,j,k],[.025,.975]);rows.append(dict(analysis=tag,cluster=cluster,model=f,metric=met,estimate=point[j,k],lower=lo,upper=hi,replicates=B))
 for a,b in itertools.combinations(range(len(names)),2):
  for k,met in enumerate(['auroc','average_precision','brier']):
   dd=draw[:,a,k]-draw[:,b,k];obs=point[a,k]-point[b,k];lo,hi=np.nanquantile(dd,[.025,.975]);p=(1+np.sum(abs(dd-obs)>=abs(obs)))/(B+1)
   pairs.append(dict(analysis=tag,model_a=names[a],model_b=names[b],metric=met,difference=obs,lower=lo,upper=hi,p_bootstrap=p))
 pp=pd.DataFrame(pairs)
 for met,g in pp.groupby('metric'):
  ix=g.sort_values('p_bootstrap').index;pp.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(pp.loc[ix,'p_bootstrap'].to_numpy()*np.arange(len(ix),0,-1)))
 pd.DataFrame(rows).to_csv(A/f'intervals_{tag}_{cluster}.csv',index=False);pp.to_csv(A/f'pairs_{tag}_{cluster}.csv',index=False)
 return draw
def domain_inputs(data):
 out={}
 for s,(d,x) in data.items():
  mean=np.nanmean(x,axis=1);missing=np.isnan(x).mean(1)
  out[s]=np.c_[mean,missing,d.age,(d.sex=='M').astype(float)]
 return out
def domain_analysis(data):
 z=domain_inputs(data);idx={s:training_indices(d) for s,(d,x) in data.items()};tr=np.concatenate([z[s][idx[s]] for s in ['mimic','eicu']]);y=np.r_[np.zeros(len(idx['mimic'])),np.ones(len(idx['eicu']))]
 est=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=42).fit(tr,y)
 preds={s:est.predict_proba(v)[:,1] for s,v in z.items()};weights={s:np.clip(p,.05,.95) if s=='mimic' else 1-np.clip(p,.05,.95) for s,p in preds.items()}
 records=[];yy=[];pp=[]
 for s,(d,x) in data.items():
  ix=np.flatnonzero(d.split=='test');w=weights[s][ix];yy.extend(np.repeat(int(s=='eicu'),len(ix)));pp.extend(preds[s][ix]);records.append(dict(source=s,n=len(ix),effective_sample_size=w.sum()**2/(w*w).sum(),fraction_domain_probability_extreme=np.mean((preds[s][ix]<.05)|(preds[s][ix]>.95)),mean_weight=w.mean()))
 pd.DataFrame(records).to_csv(A/'overlap_diagnostics.csv',index=False);(A/'domain.json').write_text(json.dumps({'held_out_auc':roc_auc_score(yy,pp),'features':'six means, six missing proportions, age, male; development-only fitting','purpose':'Descriptive overlap population, not causal correction'},indent=2))
 bal=[]
 for j,name in enumerate(['mean_'+f for f in ['hr','rr','spo2','sbp','dbp','map']]+['missing_'+f for f in ['hr','rr','spo2','sbp','dbp','map']]+['age','male']):
  vals=[]
  for s in ['mimic','eicu']:
   d=data[s][0];ix=np.flatnonzero((d.split=='test')&np.isfinite(z[s][:,j]));v=z[s][ix,j];w=weights[s][ix];vals.append((v.mean(),v.var(),np.average(v,weights=w),np.average((v-np.average(v,weights=w))**2,weights=w)))
  a,b=vals;bal.append(dict(feature=name,smd_native=(a[0]-b[0])/np.sqrt((a[1]+b[1])/2),smd_overlap=(a[2]-b[2])/np.sqrt((a[3]+b[3])/2)))
 pd.DataFrame(bal).to_csv(A/'balance.csv',index=False);return weights
def main():
 data=raw_data();weights=domain_analysis(data);rows=[];curves=[];conf=[];decision=[];localrows=[];subgroups=[];sites=[];gaps=[];neg=[]
 for source in ['mimic','eicu']:
  target='eicu' if source=='mimic' else 'mimic';sd=data[source][0];sy=sd.y.to_numpy();ci=np.flatnonzero(sd.split=='conformal')
  probs={s:{} for s in data};ovprobs={s:{} for s in data}
  for f in FAMILIES:
   z=np.load(P/f'{source}_{f}_42.npz');cp=z[source][ci];cy=sy[ci]
   for ev,(d,x) in data.items():
    ix=np.flatnonzero(d.split=='test');y=d.y.to_numpy()[ix];p=z[ev][ix];w=weights[ev][ix];probs[ev][f]=p
    stat=calibration_stats(y,p,d.patient_id.iloc[ix]);offset=logit(np.clip(p,1e-7,1-1e-7));fitoff=minimize_scalar(lambda a:np.mean(np.logaddexp(0,offset+a)-y*(offset+a)),bounds=(-10,10),method='bounded')
    fpr,tpr,_=roc_curve(y,p,drop_intermediate=False);refprev=.1;precision=np.divide(refprev*tpr,refprev*tpr+(1-refprev)*fpr,out=np.ones_like(tpr),where=(refprev*tpr+(1-refprev)*fpr)>0);ap10=np.sum(np.diff(tpr)*precision[1:])
    rows.append(dict(train_source=source,evaluation=ev,model=f,regime='native',**metrics(y,p),**stat,calibration_in_the_large=fitoff.x,observed_expected=y.mean()/p.mean(),ap_at10percent_prevalence=ap10))
    rows.append(dict(train_source=source,evaluation=ev,model=f,regime='overlap',**metrics(y,p,w)))
    good=d.core_observed.to_numpy()[ix];rows.append(dict(train_source=source,evaluation=ev,model=f,regime='core_observed',**metrics(y[good],p[good])))
    bins=pd.qcut(p,10,labels=False,duplicates='drop')
    for b,g in pd.DataFrame({'p':p,'y':y,'bin':bins}).groupby('bin'):curves.append(dict(train_source=source,evaluation=ev,model=f,bin=b,n=len(g),predicted=g.p.mean(),observed=g.y.mean()))
    for method in ['marginal_failure_baseline','class_conditional']:
     q=conformal_q(cy,cp);q0=q if method.startswith('marginal') else conformal_q(cy,cp,cls=0);q1=q if method.startswith('marginal') else conformal_q(cy,cp,cls=1)
     conf.append(dict(train_source=source,evaluation=ev,model=f,method=method,calibration_n=len(cy),calibration_events=int(cy.sum()),**coverage(y,p,q0,q1)))
    for threshold in [.01,.025,.05,.1,.2,.3]:
     alert=p>=threshold;tp=np.sum(alert&(y==1));fp=np.sum(alert&(y==0));decision.append(dict(train_source=source,evaluation=ev,model=f,threshold=threshold,net_benefit=(tp-fp*threshold/(1-threshold))/len(y),treat_all=y.mean()-(1-y.mean())*threshold/(1-threshold),alerts_per100=100*alert.mean(),sensitivity=tp/y.sum(),ppv=tp/max(alert.sum(),1)))
    for kind,values in [('sex',d.sex.iloc[ix].to_numpy()),('age',pd.cut(d.age.iloc[ix],[17,44,64,79,100],labels=['18-44','45-64','65-79','80+']).astype(str).to_numpy())]:
     for value in np.unique(values):
      mask=values==value
      if mask.sum()>=100 and y[mask].sum()>=10:subgroups.append(dict(train_source=source,evaluation=ev,model=f,group=kind,value=value,**metrics(y[mask],p[mask])))
    if ev=='eicu':
     for sid in d.site_id.iloc[ix].unique():
      mask=d.site_id.to_numpy()[ix]==sid
      if mask.sum()>=100 and y[mask].sum()>=10:sites.append(dict(train_source=source,model=f,site_id=sid,**metrics(y[mask],p[mask])))
    neg.append(dict(train_source=source,evaluation=ev,model=f,method='always_nonevent',coverage=1-y.mean(),event_coverage=0,empty_rate=0,both_rate=0))
   td=data[target][0];ty=td.y.to_numpy();ca=np.flatnonzero(td.split=='calibrate');te=np.flatnonzero(td.split=='test');tc=np.flatnonzero(td.split=='conformal');lc=calibrate(z[target][ca],ty[ca]);lp=apply_cal(lc,z[target]);localrows.append(dict(train_source=source,evaluation=target,model=f,**metrics(ty[te],lp[te]),**calibration_stats(ty[te],lp[te],td.patient_id.iloc[te])))
   q0=conformal_q(ty[tc],lp[tc],cls=0);q1=conformal_q(ty[tc],lp[tc],cls=1);conf.append(dict(train_source=source,evaluation=target,model=f,method='local_class_conditional',calibration_n=len(tc),calibration_events=int(ty[tc].sum()),**coverage(ty[te],lp[te],q0,q1)))
  draws={}
  for ev,(d,x) in data.items():
   ix=np.flatnonzero(d.split=='test');dd=d.iloc[ix].reset_index(drop=True);tag=source+'_to_'+ev
   draws[ev]=bootstrap(dd,probs[ev],tag);bootstrap(dd,probs[ev],tag+'_overlap',weights[ev][ix])
   if ev=='eicu':bootstrap(dd,probs[ev],tag,cluster='site_id')
  # Independently permute target bootstrap draw pairing, never pair patients across databases.
  perm=np.random.default_rng(5542).permutation(B)
  for j,f in enumerate(FAMILIES):
   delta=draws[target][perm,j,0]-draws[source][:,j,0];lo,hi=np.quantile(delta,[.025,.975]);r=[v for v in rows if v['train_source']==source and v['model']==f and v['regime']=='native'];vals={v['evaluation']:v['auroc'] for v in r};gaps.append(dict(train_source=source,model=f,external_minus_source_auc=vals[target]-vals[source],lower=lo,upper=hi,interpretation='Unpaired cross-population contrast, not evidence of improvement'))
  log('Evaluated '+source)
 for name,items in [('metrics',rows),('calibration_curves',curves),('conformal',conf),('decision',decision),('local_calibration',localrows),('subgroups',subgroups),('hospital_metrics',sites),('auc_gaps',gaps),('trivial_set_baseline',neg)]:pd.DataFrame(items).to_csv(A/(name+'.csv'),index=False)
if __name__=='__main__':main()
