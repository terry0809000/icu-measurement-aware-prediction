"""Outcome-independent sample audit of raw predictor timing; no label-based sampling."""
import os
import duckdb,numpy as np,pandas as pd
from prepare import ROOT,P,A
def main():
 raw=ROOT.parent/'revision_work'/'private_v9';base=__import__('pathlib').Path(os.environ.get('ICU_RAW_ROOT','D:/ards_ahrf_pfl_work_publication_v3'))
 con=duckdb.connect();con.execute("SET threads=2; SET memory_limit='4GB'");rows=[]
 for source in ['mimic','eicu']:
  d=pd.read_parquet(P/f'{source}_meta.parquet');d=d.sample(n=min(2000,len(d)),random_state=1942);con.register('sample_stays',d)
  if source=='mimic':
   path=(base/'extracted_mimic/chartevents.csv.gz').as_posix()
   q=f"""WITH f AS (SELECT s.stay_id,s.intime,TRY_CAST(v.itemid AS INT) itemid,TRY_CAST(v.valuenum AS DOUBLE) val,TRY_CAST(v.charttime AS TIMESTAMP) ctime,greatest(TRY_CAST(v.charttime AS TIMESTAMP),TRY_CAST(v.storetime AS TIMESTAMP)) atime FROM read_csv_auto('{path}',all_varchar=true) v JOIN sample_stays s ON TRY_CAST(v.stay_id AS BIGINT)=s.stay_id WHERE TRY_CAST(v.itemid AS INT) IN (220045,220210,220277,220179,220180,220181)), b AS (SELECT *,ceil(epoch(atime-intime)/3600) AS hour,CASE itemid WHEN220045 THEN'hr' WHEN220210 THEN'rr' WHEN220277 THEN'spo2' WHEN220179 THEN'sbp' WHEN220180 THEN'dbp' WHEN220181 THEN'map' END feature FROM f WHERE ctime>=intime AND epoch(atime-ctime)<=21600 AND ((itemid=220045 AND val BETWEEN20 AND250) OR(itemid=220210 AND val BETWEEN4 AND80) OR(itemid=220277 AND val BETWEEN40 AND100) OR(itemid=220179 AND val BETWEEN40 AND260) OR(itemid=220180 AND val BETWEEN20 AND180) OR(itemid=220181 AND val BETWEEN20 AND200))), last AS (SELECT stay_id,hour,feature,hour*60-max(epoch(atime-intime)/60) recency_minutes,arg_max(epoch(atime-ctime)/60,struct_pack(t:=atime,v:=val)) recording_lag_minutes FROM b WHERE hour BETWEEN18 AND23 GROUP BY1,2,3) SELECT feature,count(*) bins,avg(recency_minutes) mean_recency_minutes,median(recency_minutes) median_recency_minutes,avg(recording_lag_minutes) mean_recording_lag_minutes FROM last GROUP BY1"""
   # SQL whitespace is normalized for explicit numeric tokens.
   import re
   q=re.sub(r'\b(WHEN|BETWEEN|AND|BY)(\d)',r'\1 \2',q)
   res=con.execute(q).df();res['source']=source;res['sample_patients']=len(d);rows.extend(res.to_dict('records'))
  else:
   for table,fields in [('vitalPeriodic',[('hr','heartrate',20,250),('rr','respiration',4,80),('spo2','sao2',40,100)]),('vitalAperiodic',[('sbp','noninvasivesystolic',40,260),('dbp','noninvasivediastolic',20,180),('map','noninvasivemean',20,200)])]:
    path=(base/'extracted_eicu'/(table+'.csv.gz')).as_posix()
    branches=' UNION ALL '.join([f"SELECT stay_id,obs,'{feat}' feature,{col} val FROM f WHERE {col} BETWEEN {lo} AND {hi}" for feat,col,lo,hi in fields]);cols=','.join(f'TRY_CAST(v.{col} AS DOUBLE) {col}' for feat,col,lo,hi in fields)
    q=f"""WITH f AS (SELECT s.stay_id,TRY_CAST(v.observationoffset AS DOUBLE) obs,{cols} FROM read_csv_auto('{path}',all_varchar=true) v JOIN sample_stays s ON TRY_CAST(v.patientunitstayid AS BIGINT)=s.stay_id WHERE TRY_CAST(v.observationoffset AS DOUBLE)>1020 AND TRY_CAST(v.observationoffset AS DOUBLE)<=1380), long AS ({branches}), last AS (SELECT stay_id,ceil(obs/60) AS hour,feature,ceil(obs/60)*60-max(obs) recency_minutes FROM long GROUP BY1,2,3) SELECT feature,count(*) bins,avg(recency_minutes) mean_recency_minutes,median(recency_minutes) median_recency_minutes FROM last GROUP BY1""".replace('BY1','BY 1')
    res=con.execute(q).df();res['source']=source;res['sample_patients']=len(d);res['mean_recording_lag_minutes']=np.nan;rows.extend(res.to_dict('records'))
  print('Raw timing audited',source,flush=True)
 pd.DataFrame(rows).to_csv(A/'raw_predictor_timing.csv',index=False);con.close()
if __name__=='__main__':main()
