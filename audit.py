import json,os
import numpy as np,pandas as pd,duckdb
from scipy.special import logit
from sklearn.metrics import roc_auc_score,average_precision_score
from models import *
from statistics_core import FastMetric,conformal_q,conformal_metrics
from prepare import EX
checks=[]
def check(name,value):
 checks.append(dict(check=name,passed=bool(value)))
 if not value:raise AssertionError(name)
def main():
 data=raw_data();rng=np.random.default_rng(534)
 for s,(d,x) in data.items():
  check(s+' one stay per patient',d.patient_id.is_unique and d.stay_id.is_unique)
  check(s+' exact primary label',np.array_equal(d.y,d.death_hour.notna().astype(int)))
  check(s+' landmark risk set',(d.los>24).all() and (d.death_hour.isna()|(d.death_hour>24)).all())
  check(s+' fixed dimensions',x.shape==(len(d),6,6))
  check(s+' finite or missing inputs',not np.isinf(x).any())
  if s=='eicu':check('eICU no hospital crossover',d.groupby('site_id').split.nunique().max()==1)
  indices=training_indices(d);check(s+' source training only',(d.split.iloc[indices]=='train').all())
  original=pd.read_parquet(EX/f'{s}_cohort.parquet').set_index('stay_id').loc[d.stay_id]
  check(s+' extraction identities independent join',np.allclose(d.los,original.los) and np.allclose(d.age,original.age))
  # Reconstruct 250 selected measurement tensors independently with dictionary keys.
  mm=pd.read_parquet(EX/'mimic_measurements.parquet') if s=='mimic' else pd.concat([pd.read_parquet(EX/'eicu_periodic.parquet'),pd.read_parquet(EX/'eicu_aperiodic.parquet')])
  sample=rng.choice(len(d),250,replace=False);mm=mm[mm.stay_id.isin(d.stay_id.iloc[sample])&mm.hour.between(18,23)];lookup={(r.stay_id,r.hour,r.feature):r.val for r in mm.itertuples()}
  rebuilt=np.array([[[lookup.get((d.stay_id.iloc[i],t,f),np.nan) for f in ['hr','rr','spo2','sbp','dbp','map']] for t in range(18,24)] for i in sample]);check(s+' independent feature assembly',np.allclose(x[sample],rebuilt,equal_nan=True))
 check('equal development patients',len(training_indices(data['mimic'][0]))==len(training_indices(data['eicu'][0])))
 for v in [24,24.0001,48,48.0001]:check('secondary horizon boundary '+str(v),bool((v>24)&(v<=48))==(v in [24.0001,48]))
 y=rng.integers(0,2,300);p=rng.choice(np.linspace(.01,.99,20),300);w=rng.integers(0,5,300).astype(float);auc,ap,bs=FastMetric(y,p)(w)
 check('tied weighted AUC against sklearn',np.isclose(auc,roc_auc_score(y,p,sample_weight=w)));check('tied weighted AP against sklearn',np.isclose(ap,average_precision_score(y,p,sample_weight=w)));check('weighted Brier',np.isclose(bs,np.average((y-p)**2,weights=w)))
 # Independent literal set construction, including the exact previous reported failure mechanism.
 p=np.array([.01,.02,.05,.5,.99]);y=np.array([0,1,1,0,1]);q=.03142585192592593
 sets=[{k for k in [0,1] if (1-(v if k==1 else 1-v))<=q} for v in p]
 r=conformal_metrics(y,p,q,q)
 check('conformal set inclusion independent enumeration',np.isclose(r['coverage'],np.mean([int(a in b) for a,b in zip(y,sets)])))
 check('empty sets are not negative predictions',sets[2]==set() and sets[3]==set())
 cy=np.r_[np.zeros(97),np.ones(3)];cp=np.repeat(.02,100);q=conformal_q(cy,cp);r=conformal_metrics(cy,cp,q,q)
 check('rare event marginal zero coverage reproduced',r['event_coverage']==0 and r['coverage']==.97)
 q0=conformal_q(cy,cp,cls=0);q1=conformal_q(cy,cp,cls=1);r=conformal_metrics(cy,cp,q0,q1)
 check('class conditional small sample conservative',q1==1 and r['event_coverage']==1 and r['both_rate']==1)
 # Raw mortality status check: independently join the compressed source table.
 raw=Path(os.environ.get('ICU_RAW_ROOT','D:/ards_ahrf_pfl_work_publication_v3'));con=duckdb.connect()
 e=con.execute(f"SELECT TRY_CAST(patientunitstayid AS BIGINT) stay_id,lower(unitdischargestatus) status,TRY_CAST(unitdischargeoffset AS DOUBLE)/60 los FROM read_csv_auto('{(raw/'extracted_eicu/patient.csv.gz').as_posix()}',all_varchar=true)").df().set_index('stay_id');d=data['eicu'][0];raw_e=e.loc[d.stay_id];check('eICU primary outcome raw status',np.array_equal(d.y,(raw_e.status=='expired').astype(int)));check('eICU raw offset',np.allclose(d.los,raw_e.los))
 con.close();m=pd.read_parquet(EX/'mimic_cohort.parquet');v=m[m.death_hour.notna()];delay=v.los-v.death_hour
 pd.DataFrame([dict(recorded_icu_deaths=len(v),median_death_to_exit_hours=delay.median(),p95_death_to_exit_hours=delay.quantile(.95),known_death_before24_exit_after24=int(((m.death_hour<=24)&(m.los>24)).sum()),native_death24to48_exit_after48=int(((m.death_hour>24)&(m.death_hour<=48)&(m.los>48)).sum()))]).to_csv(A/'timestamp_audit.csv',index=False)
 for source in ['mimic','eicu']:
  for f in FAMILIES:
   for seed in [42,117,2026]:
    file=P/f'{source}_{f}_{seed}.npz'
    if not file.exists():continue
    z=np.load(file)
    for ev,(d,x) in data.items():check(f'{source}/{f}/{seed}/{ev} bounded predictions',len(z[ev])==len(d) and np.isfinite(z[ev]).all() and ((z[ev]>=0)&(z[ev]<=1)).all())
 pd.DataFrame(checks).to_csv(A/'audit_checks.csv',index=False);print('Passed',len(checks),'checks',flush=True)
if __name__=='__main__':main()
