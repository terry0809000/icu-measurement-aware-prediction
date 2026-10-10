from pathlib import Path
import sys, json, hashlib, time
import numpy as np,pandas as pd,joblib
from scipy.special import logit,expit
from scipy.optimize import minimize_scalar
from run_search import TunedTransformer, HERE, ROOT, PRIVATE, RESULTS, SEEDS
from models import raw_data,transform,predict,metrics,calibrate,apply_cal,FAMILIES
from statistics_core import FastMetric,calibration_stats,conformal_q,conformal_metrics

def citl(y,p):
    z=logit(np.clip(p,1e-7,1-1e-7))
    return float(minimize_scalar(lambda b:np.mean(np.logaddexp(0,z+b)-y*(z+b)),bounds=(-10,10),method='bounded').x)

def coverage(y,p,q0,q1):
    out=conformal_metrics(y,p,q0,q1)
    for key,n in [('coverage',len(y)),('event_coverage',int(y.sum())),('nonevent_coverage',int((y==0).sum()))]:
        rate=out[key];z=1.959963984540054;den=1+z*z/n
        center=(rate+z*z/(2*n))/den;half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
        out[key+'_wilson_lower']=max(0,center-half);out[key+'_wilson_upper']=min(1,center+half)
    return out

def evaluate():
    lock=RESULTS/'locked_selection.json'
    assert hashlib.sha256(lock.read_bytes()).hexdigest()==(RESULTS/'locked_selection.sha256').read_text()
    manifest=json.loads(lock.read_text()); choices=manifest['selection']
    assert not manifest['test_evaluated']
    data=raw_data();rows=[];sets=[];updates=[];allpred={}
    for source in ['mimic','eicu']:
        sd=data[source][0];sy=sd.y.to_numpy();ca=np.flatnonzero(sd.split=='calibrate');co=np.flatnonzero(sd.split=='conformal')
        prep=json.loads((PRIVATE/f'{source}_preprocessing.json').read_text())
        xx={s:transform(x,d,prep) for s,(d,x) in data.items()}
        for f in FAMILIES:
            j=choices[source][f]['candidate']
            for seed in SEEDS:
                est=joblib.load(PRIVATE/f'{source}_{f}_{j}_{seed}.joblib')
                raw={s:predict(est,x,f) for s,x in xx.items()}
                cal=calibrate(raw[source][ca],sy[ca]); pred={s:apply_cal(cal,p) for s,p in raw.items()}
                assert cal.coef_[0,0]>0, 'Calibration reverses ranking; investigate without test-based replacement'
                np.savez_compressed(PRIVATE/f'{source}_{f}_{seed}_predictions.npz',**pred)
                for target,(d,x) in data.items():
                    te=np.flatnonzero(d.split=='test');y=d.y.to_numpy()[te];p=pred[target][te]
                    assert len(p)==len(te) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    result=dict(train_source=source,evaluation=target,model=f,seed=seed,source_selected=(f==choices[source]['selected_family']),**metrics(y,p),**calibration_stats(y,p),calibration_in_the_large=citl(y,p))
                    rows.append(result)
                    if seed!=42:continue
                    allpred[(source,target,f)]=p
                    for method in ['marginal','class_conditional']:
                        q0=conformal_q(sy[co],pred[source][co],cls=0 if method=='class_conditional' else None)
                        q1=conformal_q(sy[co],pred[source][co],cls=1 if method=='class_conditional' else None)
                        sets.append(dict(train_source=source,evaluation=target,model=f,method=method,**coverage(y,p,q0,q1)))
                    if target!=source:
                        tc=np.flatnonzero(d.split=='calibrate');tf=np.flatnonzero(d.split=='conformal');dy=d.y.to_numpy()
                        localcal=calibrate(raw[target][tc],dy[tc]);lp=apply_cal(localcal,raw[target])
                        updates.append(dict(train_source=source,evaluation=target,model=f,**metrics(y,lp[te]),**calibration_stats(y,lp[te]),calibration_in_the_large=citl(y,lp[te])))
                        q0=conformal_q(dy[tf],lp[tf],cls=0);q1=conformal_q(dy[tf],lp[tf],cls=1)
                        sets.append(dict(train_source=source,evaluation=target,model=f,method='target_updated_class_conditional',**coverage(y,lp[te],q0,q1)))
                print('evaluated',source,f,seed,flush=True)
    pd.DataFrame(rows).to_csv(RESULTS/'test_metrics_all_seeds.csv',index=False)
    pd.DataFrame(sets).to_csv(RESULTS/'conformal.csv',index=False)
    pd.DataFrame(updates).to_csv(RESULTS/'local_calibration.csv',index=False)
    intervals=[];differences=[]
    for target,(d,x) in data.items():
        te=np.flatnonzero(d.split=='test');y=d.y.to_numpy()[te];n=len(te)
        keys=[];fm=[]
        for source in ['mimic','eicu']:
            for f in FAMILIES:
                keys.append(('extension',source,f));fm.append(FastMetric(y,allpred[(source,target,f)]))
                old=np.load(ROOT/'private'/f'{source}_{f}_42.npz')[target][te]
                keys.append(('v10',source,f));fm.append(FastMetric(y,old))
        for unit in (['patient','hospital'] if target=='eicu' else ['patient']):
            rng=np.random.default_rng(71209 if unit=='patient' else 71210)
            if unit=='hospital': codes,sites=pd.factorize(d.site_id.iloc[te]);ng=len(sites)
            reps=[]
            for b in range(1000):
                if unit=='patient':w=np.bincount(rng.integers(0,n,n),minlength=n)
                else:w=np.bincount(rng.integers(0,ng,ng),minlength=ng)[codes]
                reps.append([f(w) for f in fm])
            reps=np.array(reps)
            for k,key in enumerate(keys):
                if key[0]!='extension':continue
                point=fm[k](np.ones(n))
                for z,name in enumerate(['auroc','average_precision','brier']):
                    lo,hi=np.nanquantile(reps[:,k,z],[.025,.975]);intervals.append(dict(source=key[1],evaluation=target,model=key[2],resampling=unit,metric=name,estimate=point[z],lower=lo,upper=hi))
                oldk=keys.index(('v10',key[1],key[2]))
                for z,name in enumerate(['auroc','average_precision','brier']):
                    diff=reps[:,k,z]-reps[:,oldk,z];lo,hi=np.nanquantile(diff,[.025,.975]);differences.append(dict(source=key[1],evaluation=target,model=key[2],comparison='same_family_v10',resampling=unit,metric=name,difference=point[z]-fm[oldk](np.ones(n))[z],lower=lo,upper=hi))
            for source in ['mimic','eicu']:
                selected=choices[source]['selected_family'];oldselected=json.loads((ROOT/'results'/f'{source}_selected.json').read_text())['model']
                a=keys.index(('extension',source,selected));b=keys.index(('v10',source,oldselected))
                for z,name in enumerate(['auroc','average_precision','brier']):
                    lo,hi=np.nanquantile(reps[:,a,z]-reps[:,b,z],[.025,.975]);differences.append(dict(source=source,evaluation=target,model=selected,comparison='source_selected_v10',resampling=unit,metric=name,difference=fm[a](np.ones(n))[z]-fm[b](np.ones(n))[z],lower=lo,upper=hi))
            print('bootstrap complete',target,unit,flush=True)
    pd.DataFrame(intervals).to_csv(RESULTS/'test_intervals.csv',index=False)
    pd.DataFrame(differences).to_csv(RESULTS/'paired_v10_differences.csv',index=False)
    report={'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'locked_manifest_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'all_test_predictions_finite_and_bounded':True,'all_probability_calibrators_monotone_increasing':True,'test_used_for_selection':False,'bootstrap_replicates':1000,'search_candidates_per_family_per_source':16,'confirmation_candidates_per_family_per_source':2,'confirmation_seeds':SEEDS}
    (RESULTS/'evaluation_complete.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': evaluate()
