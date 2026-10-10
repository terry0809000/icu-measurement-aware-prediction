from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from run_search import HERE,ROOT,RESULTS,PRIVATE,FAMILIES,SEEDS
from models import raw_data,training_indices
from prepare import h

def main():
    checks=[]
    def check(name,value):
        checks.append({'check':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    lockpath=RESULTS/'locked_selection.json';lock=json.loads(lockpath.read_text());choice=lock['selection']
    check('selection manifest hash unchanged',hashlib.sha256(lockpath.read_bytes()).hexdigest()==(RESULTS/'locked_selection.sha256').read_text())
    check('protocol unchanged since selection',hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest()==lock['protocol_sha256'])
    check('search implementation unchanged since selection',hashlib.sha256((HERE/'run_search.py').read_bytes()).hexdigest()==lock['code_sha256'])
    allcounts=[]
    for s,(d,x) in raw_data().items():
        tr=training_indices(d);g=d.site_id if s=='eicu' else d.patient_id
        inner=np.array([h(v,'v11-inner-screen-20260919')<.2 for v in g.iloc[tr]])
        a=tr[~inner];b=tr[inner]
        check(s+' train and screen disjoint',len(set(a)&set(b))==0)
        check(s+' combined inner sample equals original matched train',set(a)|set(b)==set(tr))
        check(s+' final training budget unchanged',len(tr)==27437)
        check(s+' patient unique',d.patient_id.is_unique)
        check(s+' train screening never uses held out roles',(d.split.iloc[np.r_[a,b]]=='train').all())
        if s=='eicu':
            check('eICU inner hospital split disjoint',not set(d.site_id.iloc[a])&set(d.site_id.iloc[b]))
            check('eICU outer hospital roles disjoint',d.groupby('site_id').split.nunique().max()==1)
        for f in FAMILIES:
            screen=pd.read_csv(RESULTS/f'{s}_{f}_screen.csv');confirm=pd.read_csv(RESULTS/f'{s}_{f}_confirm.csv')
            check(s+'/'+f+' all sixteen candidates retained',len(screen)==16 and set(screen.candidate)==set(range(16)))
            shortlist=screen.sort_values(['auroc','average_precision','candidate'],ascending=[False,False,True]).candidate.iloc[:2].tolist()
            check(s+'/'+f+' correct shortlist',set(confirm.candidate)==set(shortlist))
            check(s+'/'+f+' all six confirmation fits retained',len(confirm)==6 and not confirm.duplicated(['candidate','seed']).any() and all(set(z.seed)==set(SEEDS) for _,z in confirm.groupby('candidate')))
            rank=confirm.groupby('candidate').agg(auc=('auroc','mean'),ap=('average_precision','mean')).reset_index().sort_values(['auc','ap','candidate'],ascending=[False,False,True])
            check(s+'/'+f+' winner follows validation rule',int(rank.iloc[0].candidate)==choice[s][f]['candidate'])
            check(s+'/'+f+' recorded validation AUROC agrees',np.isclose(rank.iloc[0].auc,choice[s][f]['validation_auc']))
            check(s+'/'+f+' matched confirmation budget',(confirm.train_n==27437).all())
            for seed in SEEDS:
                pp=PRIVATE/f'{s}_{f}_{seed}_predictions.npz'
                check(f'{s}/{f}/{seed} prediction file created after selection lock',pp.stat().st_mtime>=lockpath.stat().st_mtime)
            allcounts.append((len(screen),len(confirm)))
        family=max(FAMILIES,key=lambda f:(choice[s][f]['validation_auc'],choice[s][f]['validation_ap'],-FAMILIES.index(f)))
        check(s+' family chosen without test metrics',family==choice[s]['selected_family'])
    m=pd.read_csv(RESULTS/'test_metrics_all_seeds.csv')
    check('all forty eight model seed population evaluations retained',len(m)==48 and not m.duplicated(['train_source','evaluation','model','seed']).any())
    check('all AUROC and AP estimates finite',np.isfinite(m[['auroc','average_precision']]).all().all())
    check('exactly 128 screening and 48 confirmation fits',np.array(allcounts).sum(axis=0).tolist()==[128,48])
    ci=pd.read_csv(RESULTS/'test_intervals.csv')
    check('patient and hospital bootstrap intervals present',set(ci.resampling)=={'patient','hospital'})
    check('hospital intervals restricted to eICU',(ci[ci.resampling=='hospital'].evaluation=='eicu').all())
    pd.DataFrame(checks).to_csv(RESULTS/'verification_checks.csv',index=False)
    print('Passed',len(checks),'extension checks')

if __name__=='__main__':main()
