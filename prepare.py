from pathlib import Path
import os,hashlib,json
import numpy as np,pandas as pd,duckdb
ROOT=Path(__file__).resolve().parent;P=ROOT/'private';A=ROOT/'results'
P.mkdir(exist_ok=True);A.mkdir(exist_ok=True)
EX=Path(os.environ.get('ICU_EXTRACT_ROOT',str(ROOT.parent/'revision_work'/'private_v9')))
FEATURES=['hr','rr','spo2','sbp','dbp','map']
def h(x,salt):return int(hashlib.sha256((salt+'|'+str(x)).encode()).hexdigest()[:12],16)/16**12
def role(x):return 'train' if x<.5 else 'tune' if x<.6 else 'calibrate' if x<.7 else 'conformal' if x<.8 else 'test'
def main():
 flows=[];coh=[];missing=[]
 for source in ['mimic','eicu']:
  c=pd.read_parquet(EX/f'{source}_cohort.parquet');flows.append(dict(source=source,stage='adult_stays_gt12',stays=len(c),patients=c.patient_id.nunique()))
  c=c[(c.los>24)&(c.death_hour.isna()|(c.death_hour>24))].copy()
  if source=='eicu':
   raw=Path(os.environ.get('ICU_RAW_ROOT','D:/ards_ahrf_pfl_work_publication_v3'))/'extracted_eicu'/'patient.csv.gz'
   con=duckdb.connect();status=con.execute(f"SELECT TRY_CAST(patientunitstayid AS BIGINT) stay_id,lower(unitdischargestatus) discharge_status FROM read_csv_auto('{raw.as_posix()}',all_varchar=true)").df();con.close()
   c=c.merge(status,on='stay_id',validate='one_to_one');c=c[c.discharge_status.isin(['alive','expired'])].copy()
  flows.append(dict(source=source,stage='eligible_at24_known_status',stays=len(c),patients=c.patient_id.nunique()))
  c['patient_id']=source+':'+c.patient_id.astype(str)
  c['split']=[role(h(x,'v10-hospital')) for x in c.site_id] if source=='eicu' else [role(h(x,'v10-patient')) for x in c.patient_id]
  if source=='eicu':
   mixed=c.groupby('patient_id').split.nunique();c=c[~c.patient_id.isin(mixed[mixed>1].index)].copy()
   flows.append(dict(source=source,stage='hospital_roles_no_cross_role_patient',stays=len(c),patients=c.patient_id.nunique()))
  c['stay_rank']=[h(x,'v10-index-stay') for x in c.stay_id]
  c=c.sort_values('stay_rank').drop_duplicates('patient_id').sort_values('patient_id').reset_index(drop=True)
  c['y']=c.death_hour.notna().astype('int8');c['hour']=24
  m=pd.read_parquet(EX/'mimic_measurements.parquet') if source=='mimic' else pd.concat([pd.read_parquet(EX/'eicu_periodic.parquet'),pd.read_parquet(EX/'eicu_aperiodic.parquet')])
  m=m[m.stay_id.isin(c.stay_id)&m.hour.between(18,23)]
  wide=m.pivot(index=['stay_id','hour'],columns='feature',values='val').reindex(columns=FEATURES)
  x=np.stack([wide.reindex(pd.MultiIndex.from_arrays([c.stay_id,np.repeat(t,len(c))],names=['stay_id','hour'])).to_numpy('float32') for t in range(18,24)],axis=1)
  c['missing_fraction']=np.isnan(x).mean((1,2));c['core_observed']=(np.isfinite(x[:,:,:3]).sum(1)>=3).all(1)
  c.to_parquet(P/f'{source}_meta.parquet',index=False);np.save(P/f'{source}_X.npy',x)
  for split,g in c.groupby('split'):
   coh.append(dict(source=source,split=split,patients=len(g),events=int(g.y.sum()),prevalence=g.y.mean(),sites=g.site_id.nunique(),median_age=g.age.median(),female_fraction=(g.sex=='F').mean(),missing_fraction=g.missing_fraction.mean(),early_live_discharge=int(((g.los<48)&g.death_hour.isna()).sum())))
   xx=x[g.index]
   for j,f in enumerate(FEATURES):missing.append(dict(source=source,split=split,feature=f,missing_fraction=np.isnan(xx[:,:,j]).mean()))
  density=m.groupby('feature').measurement_count.agg(['count','mean','median']).reset_index();density['source']=source;density.to_csv(A/f'{source}_density.csv',index=False)
  assert c.patient_id.is_unique and c.stay_id.is_unique
  assert ((c.death_hour.isna())|(c.death_hour>24)).all()
  print(source,len(c),c.y.sum(),flush=True)
 pd.DataFrame(coh).to_csv(A/'cohorts.csv',index=False);pd.DataFrame(flows).to_csv(A/'flow.csv',index=False);pd.DataFrame(missing).to_csv(A/'missingness.csv',index=False)
 n=int(min(r['patients'] for r in coh if r['split']=='train'))
 (A/'design.json').write_text(json.dumps({'train_n_each_source':n,'protocol_sha256':hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
