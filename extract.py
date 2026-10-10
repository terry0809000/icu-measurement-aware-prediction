"""Raw-data harmonisation. Local restricted outputs are never included in submission ZIP."""
from pathlib import Path
import json,time,hashlib,sys,os
import duckdb

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'raw_extract'; OUT.mkdir(exist_ok=True)
RAW=Path(os.environ.get('ICU_RAW_ROOT','D:/ards_ahrf_pfl_work_publication_v3'))
con=duckdb.connect(str(OUT/'extract.duckdb'))
con.execute("SET threads=4; SET memory_limit='10GB'; SET preserve_insertion_order=false")
def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def sql(s): return con.execute(s)
def view(name,folder,file):
 p=RAW/folder/(file+'.csv.gz')
 sql(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{p.as_posix()}', all_varchar=true, sample_size=20000)")
def export(q,name):
 p=OUT/(name+'.parquet')
 if p.exists(): log('Cached '+name); return
 log('Building '+name)
 sql(f"COPY ({q}) TO '{p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
 log('Finished '+name)
for table in ['patients','icustays','admissions','chartevents','d_items']:
 view('m_'+table,'extracted_mimic',table)
for table in ['patient','vitalPeriodic','vitalAperiodic']:
 view('e_'+table.lower(),'extracted_eicu',table)
export("""SELECT CAST(i.stay_id AS BIGINT) stay_id, i.subject_id patient_id, 0 site_id,
 least(89, CAST(p.anchor_age AS INT)+year(CAST(i.intime AS TIMESTAMP))-CAST(p.anchor_year AS INT)) age,
 p.gender sex, a.race, p.anchor_year_group, CAST(i.intime AS TIMESTAMP) intime, CAST(i.outtime AS TIMESTAMP) outtime,
 epoch(CAST(i.outtime AS TIMESTAMP)-CAST(i.intime AS TIMESTAMP))/3600 los,
 CASE WHEN CAST(a.deathtime AS TIMESTAMP) BETWEEN CAST(i.intime AS TIMESTAMP) AND CAST(i.outtime AS TIMESTAMP)
 THEN epoch(CAST(a.deathtime AS TIMESTAMP)-CAST(i.intime AS TIMESTAMP))/3600 END death_hour
 FROM m_icustays i JOIN m_patients p USING(subject_id) JOIN m_admissions a USING(hadm_id)
 WHERE CAST(p.anchor_age AS INT)+year(CAST(i.intime AS TIMESTAMP))-CAST(p.anchor_year AS INT)>=18
 AND epoch(CAST(i.outtime AS TIMESTAMP)-CAST(i.intime AS TIMESTAMP))>12*3600""",'mimic_cohort')
export("""SELECT CAST(patientunitstayid AS BIGINT) stay_id, uniquepid patient_id, CAST(hospitalid AS INT) site_id,
 least(89, CASE WHEN age LIKE '%>%' THEN 90 ELSE TRY_CAST(age AS INT) END) age,
 CASE WHEN lower(gender)='male' THEN 'M' WHEN lower(gender)='female' THEN 'F' ELSE 'Unknown' END sex,
 ethnicity race, CAST(unitdischargeoffset AS DOUBLE)/60 los,
 CASE WHEN lower(unitdischargestatus)='expired' THEN CAST(unitdischargeoffset AS DOUBLE)/60 END death_hour
 FROM e_patient WHERE (CASE WHEN age LIKE '%>%' THEN 90 ELSE TRY_CAST(age AS INT) END)>=18
 AND CAST(unitdischargeoffset AS DOUBLE)>12*60 AND uniquepid IS NOT NULL""",'eicu_cohort')
for source in ['mimic','eicu']:
 sql(f"CREATE OR REPLACE VIEW {source}_cohort AS SELECT * FROM read_parquet('{(OUT/(source+'_cohort.parquet')).as_posix()}')")
export("""SELECT itemid,label,unitname FROM m_d_items WHERE TRY_CAST(itemid AS INT) IN
 (220045,220210,220277,220179,220180,220181,221906,221289,221749,222315,221662)""",'mimic_item_audit')

# Last available measurement, not the most extreme of unequal-density streams.
export("""WITH filtered AS (
 SELECT c.stay_id, TRY_CAST(e.itemid AS INT) itemid, TRY_CAST(e.valuenum AS DOUBLE) val,
 greatest(CAST(e.charttime AS TIMESTAMP),CAST(e.storetime AS TIMESTAMP)) atime,
 c.intime, TRY_CAST(e.caregiver_id AS BIGINT) record_tie
 FROM m_chartevents e JOIN mimic_cohort c ON TRY_CAST(e.stay_id AS BIGINT)=c.stay_id
 WHERE TRY_CAST(e.itemid AS INT) IN (220045,220210,220277,220179,220180,220181)
 AND CAST(e.charttime AS TIMESTAMP)>=c.intime
 AND greatest(CAST(e.charttime AS TIMESTAMP),CAST(e.storetime AS TIMESTAMP))<=c.intime+INTERVAL '72 hours'
 AND epoch(greatest(CAST(e.charttime AS TIMESTAMP),CAST(e.storetime AS TIMESTAMP))-CAST(e.charttime AS TIMESTAMP))<=21600
 ), mapped AS (SELECT *, CAST(ceil(epoch(atime-intime)/3600) AS INT) AS hour,
 CASE itemid WHEN 220045 THEN 'hr' WHEN 220210 THEN 'rr' WHEN 220277 THEN 'spo2'
 WHEN 220179 THEN 'sbp' WHEN 220180 THEN 'dbp' WHEN 220181 THEN 'map' END feature
 FROM filtered WHERE (itemid=220045 AND val BETWEEN 20 AND 250) OR (itemid=220210 AND val BETWEEN 4 AND 80)
 OR (itemid=220277 AND val BETWEEN 40 AND 100) OR (itemid=220179 AND val BETWEEN 40 AND 260)
 OR (itemid=220180 AND val BETWEEN 20 AND 180) OR (itemid=220181 AND val BETWEEN 20 AND 200))
 SELECT stay_id,hour,feature,arg_max(val,struct_pack(t:=atime,v:=val)) val, count(*) measurement_count
 FROM mapped WHERE hour BETWEEN 1 AND 72 GROUP BY 1,2,3""",'mimic_measurements')
export("""WITH f AS (SELECT c.stay_id,TRY_CAST(v.observationoffset AS DOUBLE) obs_min,
 TRY_CAST(v.heartrate AS DOUBLE) hr, TRY_CAST(v.respiration AS DOUBLE) rr,TRY_CAST(v.sao2 AS DOUBLE) spo2
 FROM e_vitalperiodic v JOIN eicu_cohort c ON TRY_CAST(v.patientunitstayid AS BIGINT)=c.stay_id
 WHERE TRY_CAST(v.observationoffset AS DOUBLE)>0 AND TRY_CAST(v.observationoffset AS DOUBLE)<=4320),
 long AS (SELECT stay_id,obs_min,'hr' feature,hr val FROM f WHERE hr BETWEEN 20 AND 250
 UNION ALL SELECT stay_id,obs_min,'rr',rr FROM f WHERE rr BETWEEN 4 AND 80
 UNION ALL SELECT stay_id,obs_min,'spo2',spo2 FROM f WHERE spo2 BETWEEN 40 AND 100)
 SELECT stay_id,CAST(ceil(obs_min/60) AS INT) AS hour,feature,arg_max(val,struct_pack(t:=obs_min,v:=val)) val,count(*) measurement_count
 FROM long GROUP BY 1,2,3""",'eicu_periodic')
export("""WITH f AS (SELECT c.stay_id,TRY_CAST(v.observationoffset AS DOUBLE) obs_min,
 TRY_CAST(v.noninvasivesystolic AS DOUBLE) sbp,TRY_CAST(v.noninvasivediastolic AS DOUBLE) dbp,TRY_CAST(v.noninvasivemean AS DOUBLE) AS map
 FROM e_vitalaperiodic v JOIN eicu_cohort c ON TRY_CAST(v.patientunitstayid AS BIGINT)=c.stay_id
 WHERE TRY_CAST(v.observationoffset AS DOUBLE)>0 AND TRY_CAST(v.observationoffset AS DOUBLE)<=4320),
 long AS (SELECT stay_id,obs_min,'sbp' feature,sbp val FROM f WHERE sbp BETWEEN 40 AND 260
 UNION ALL SELECT stay_id,obs_min,'dbp',dbp FROM f WHERE dbp BETWEEN 20 AND 180
 UNION ALL SELECT stay_id,obs_min,'map',map FROM f WHERE map BETWEEN 20 AND 200)
 SELECT stay_id,CAST(ceil(obs_min/60) AS INT) AS hour,feature,arg_max(val,struct_pack(t:=obs_min,v:=val)) val,count(*) measurement_count
 FROM long GROUP BY 1,2,3""",'eicu_aperiodic')
manifest={'protocol_sha256':hashlib.sha256((ROOT/'protocol_at_training.md').read_bytes()).hexdigest(), 'source_root':str(RAW),'outputs':[]}
for p in OUT.glob('*.parquet'):
 manifest['outputs'].append({'file':p.name,'bytes':p.stat().st_size,'rows':sql(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0]})
(OUT/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2))
log('Extraction complete')



