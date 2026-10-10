from pathlib import Path
import os, sys, json, time, hashlib, itertools, gc, copy
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd, torch, joblib
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from models import raw_data,training_indices,transform,metrics,calibrate,apply_cal,predict,CONFIGS,FAMILIES
from prepare import h
torch.set_num_threads(4);threadpool_limits(4)
PRIVATE=HERE/'private';RESULTS=HERE/'results'
PRIVATE.mkdir(exist_ok=True);RESULTS.mkdir(exist_ok=True)
SEEDS=[42,117,2026]

def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def space():
    grids={}
    grids['logistic']=[{'C':x,'class_weight':None} for x in [.001,.003,.01,.03,.1,.3,1,10]]
    grids['logistic'] += [{'C':x,'class_weight':None} for x in [.00001,.0001,3,100]]
    grids['logistic'] += [{'C':x,'class_weight':'balanced'} for x in [.001,.01,.1,1]]
    grids['hist_boost']=[dict(c,iterations=200,l2=1,min_leaf=30) for c in CONFIGS['hist_boost']]
    grids['hist_boost'] += [dict(leaves=l,lr=r,iterations=n,l2=z,min_leaf=m) for l,r,n,z,m in [(7,.05,400,10,50),(15,.05,400,10,50),(31,.03,500,10,50),(31,.05,400,5,100),(63,.03,400,10,100),(15,.1,300,0,20),(7,.1,500,1,20),(31,.01,600,1,30)]]
    grids['extra_trees']=[dict(c,trees=300,depth=None) for c in CONFIGS['extra_trees']]
    grids['extra_trees'] += [dict(leaf=l,features=f,trees=400,depth=d) for l,f,d in [(1,.5,None),(2,1.,None),(5,.25,None),(10,.75,None),(10,1.,12),(20,.75,16),(30,1.,None),(5,1.,8)]]
    grids['transformer']=[dict(c,epochs=30,dropout=.1,weight_decay=.001) for c in CONFIGS['transformer']]
    grids['transformer'] += [dict(d=d,layers=l,lr=r,epochs=e,dropout=drop,weight_decay=w) for d,l,r,e,drop,w in [(16,1,.003,30,.1,.001),(32,1,.003,30,.1,.001),(32,1,.001,45,.2,.01),(32,2,.001,45,.2,.01),(64,1,.001,30,.1,.001),(64,1,.0003,45,.2,.01),(16,2,.001,45,.0,.0001),(32,1,.0003,60,.1,.001)]]
    assert all(len(v)==16 for v in grids.values())
    return grids

class TunedTransformer(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['d']
        self.embed=nn.Linear(12,d); self.position=nn.Parameter(torch.zeros(1,6,d)); self.static=nn.Linear(4,d)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,d*2,c['dropout'],batch_first=True,activation='gelu'),c['layers'],enable_nested_tensor=False)
        self.out=nn.Linear(d,1)
    def forward(self,x):
        return self.out(self.encoder(self.embed(x[:,:72].reshape(-1,6,12))+self.position).mean(1)+self.static(x[:,72:])).squeeze(1)

def fit(f,c,seed,x,y):
    np.random.seed(seed);torch.manual_seed(seed)
    if f=='logistic': m=LogisticRegression(C=c['C'],class_weight=c['class_weight'],max_iter=2000,random_state=seed)
    elif f=='hist_boost': m=HistGradientBoostingClassifier(max_iter=c['iterations'],learning_rate=c['lr'],max_leaf_nodes=c['leaves'],l2_regularization=c['l2'],min_samples_leaf=c['min_leaf'],early_stopping=False,random_state=seed)
    elif f=='extra_trees': m=ExtraTreesClassifier(n_estimators=c['trees'],min_samples_leaf=c['leaf'],max_features=c['features'],max_depth=c['depth'],n_jobs=4,random_state=seed)
    else:
        m=TunedTransformer(c);m.out.bias.data.fill_(float(logit(y.mean())))
        opt=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y.astype('float32'))),batch_size=1024,shuffle=True)
        for ep in range(c['epochs']):
            m.train()
            for xb,yb in loader:
                opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(m(xb),yb);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        return m
    m.fit(x,y);return m

def preprocessing(x,ix):
    med=np.nan_to_num(np.nanmedian(x[ix],axis=(0,1)));v=np.where(np.isnan(x[ix]),med,x[ix])
    return {'median':med.tolist(),'mean':v.mean((0,1)).tolist(),'std':np.maximum(v.std((0,1)),1e-3).tolist()}

def main():
    grids=space(); gridpath=HERE/'search_space.json'
    serialized=json.dumps(grids,indent=2)
    if gridpath.exists(): assert gridpath.read_text()==serialized
    else: gridpath.write_text(serialized)
    protocol_hash=hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest()
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (RESULTS/'started.json').write_text(json.dumps({'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'protocol_sha256':protocol_hash,'code_sha256':code_hash},indent=2))
    data=raw_data();checks=[];counts=[];chosen={}
    for source,(d,x) in data.items():
        tr=training_indices(d); tu=np.flatnonzero(d.split=='tune')
        grouping=d.site_id if source=='eicu' else d.patient_id
        inner=np.array([h(v,'v11-inner-screen-20260919')<.2 for v in grouping.iloc[tr]])
        screen=tr[inner];train=tr[~inner]
        for name,ix in [('inner_train',train),('screen',screen),('confirm',tu),('final_train',tr)]:
            assert d.y.iloc[ix].nunique()==2
            counts.append(dict(source=source,role=name,n=len(ix),events=int(d.y.iloc[ix].sum()),hospitals=d.site_id.iloc[ix].nunique()))
        assert not set(train)&set(screen) and not set(tr)&set(tu)
        assert (d.split.iloc[tr]=='train').all() and (d.split.iloc[tu]=='tune').all()
        assert d.patient_id.is_unique
        if source=='eicu':
            assert not set(d.site_id.iloc[train])&set(d.site_id.iloc[screen])
            assert d.groupby('site_id').split.nunique().max()==1
        checks.append({'source':source,'patient_and_role_separation':True,'hospital_separation_if_eicu':True,'training_only_preprocessing':True})
        prep=preprocessing(x,train); px=transform(x,d,prep); y=d.y.to_numpy()
        fullprep=preprocessing(x,tr); fullx=transform(x,d,fullprep)
        (PRIVATE/f'{source}_preprocessing.json').write_text(json.dumps(fullprep))
        chosen[source]={}
        for f in FAMILIES:
            rows=[];path=RESULTS/f'{source}_{f}_screen.csv'
            prior=pd.read_csv(path).to_dict('records') if path.exists() else []
            done={int(r['candidate']) for r in prior};rows.extend(prior)
            for j,c in enumerate(grids[f]):
                if j in done: continue
                start=time.perf_counter();log(f'{source} {f} screen {j+1}/16')
                est=fit(f,c,42,px[train],y[train]);mt=metrics(y[screen],predict(est,px[screen],f))
                rows.append(dict(source=source,model=f,candidate=j,params=json.dumps(c),train_n=len(train),seconds=time.perf_counter()-start,**mt))
                pd.DataFrame(rows).to_csv(path,index=False);del est;gc.collect()
            ranking=pd.DataFrame(rows).sort_values(['auroc','average_precision','candidate'],ascending=[False,False,True])
            finalists=ranking.candidate.iloc[:2].astype(int).tolist();confirm=[]
            cp=RESULTS/f'{source}_{f}_confirm.csv'
            prior=pd.read_csv(cp).to_dict('records') if cp.exists() else []
            done={(int(r['candidate']),int(r['seed'])) for r in prior};confirm.extend(prior)
            for j in finalists:
                for seed in SEEDS:
                    if (j,seed) in done:continue
                    log(f'{source} {f} confirm candidate {j} seed {seed}');start=time.perf_counter()
                    est=fit(f,grids[f][j],seed,fullx[tr],y[tr]);mt=metrics(y[tu],predict(est,fullx[tu],f));mttr=metrics(y[tr],predict(est,fullx[tr],f))
                    confirm.append(dict(source=source,model=f,candidate=j,seed=seed,params=json.dumps(grids[f][j]),train_n=len(tr),train_auc=mttr['auroc'],train_ap=mttr['average_precision'],seconds=time.perf_counter()-start,**mt))
                    joblib.dump(est,PRIVATE/f'{source}_{f}_{j}_{seed}.joblib');pd.DataFrame(confirm).to_csv(cp,index=False)
                    del est;gc.collect()
            summary=pd.DataFrame(confirm).groupby('candidate').agg(mean_auc=('auroc','mean'),mean_ap=('average_precision','mean'),seed_auc_sd=('auroc','std')).reset_index().sort_values(['mean_auc','mean_ap','candidate'],ascending=[False,False,True])
            winner=summary.iloc[0];j=int(winner.candidate)
            chosen[source][f]=dict(candidate=j,params=grids[f][j],validation_auc=float(winner.mean_auc),validation_ap=float(winner.mean_ap),seed_auc_sd=float(winner.seed_auc_sd))
            (RESULTS/f'{source}_{f}_chosen.json').write_text(json.dumps(chosen[source][f],indent=2))
        chosen[source]['selected_family']=max(FAMILIES,key=lambda f:(chosen[source][f]['validation_auc'],chosen[source][f]['validation_ap'],-FAMILIES.index(f)))
    pd.DataFrame(counts).to_csv(RESULTS/'partition_counts.csv',index=False)
    (RESULTS/'selection_checks.json').write_text(json.dumps(checks,indent=2))
    manifest={'locked_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'protocol_sha256':protocol_hash,'code_sha256':code_hash,'selection':chosen,'primary_seed':42,'test_evaluated':False}
    path=RESULTS/'locked_selection.json';path.write_text(json.dumps(manifest,indent=2))
    (RESULTS/'locked_selection.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    log('ALL SELECTIONS LOCKED. No extension test predictions have been computed.')

if __name__=='__main__': main()
