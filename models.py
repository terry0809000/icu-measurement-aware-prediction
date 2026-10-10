from pathlib import Path
import os,time,json,hashlib,gc,itertools
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np,pandas as pd,joblib,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from threadpoolctl import threadpool_limits
from prepare import ROOT,P,A,h
torch.set_num_threads(4);threadpool_limits(4)
FAMILIES=['logistic','hist_boost','extra_trees','transformer']
CONFIGS={'logistic':[{'C':v} for v in [.001,.003,.01,.03,.1,.3,1,10]],
 'hist_boost':[{'leaves':v,'lr':lr} for v in [7,15,31,63] for lr in [.03,.1]],
 'extra_trees':[{'leaf':v,'features':f} for v in [5,20,50,100] for f in [.5,1.]],
 'transformer':[{'d':d,'layers':n,'lr':lr} for d in [16,32] for n in [1,2] for lr in [.001,.0003]]}
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def metrics(y,p,w=None):
 p=np.clip(p,1e-7,1-1e-7);prev=np.average(y,weights=w)
 return dict(n=len(y),events=int(y.sum()),prevalence=float(prev),auroc=float(roc_auc_score(y,p,sample_weight=w)) if 0<prev<1 else np.nan,average_precision=float(average_precision_score(y,p,sample_weight=w)) if prev>0 else np.nan,brier=float(np.average((y-p)**2,weights=w)),log_loss=float(log_loss(y,p,sample_weight=w,labels=[0,1])),mean_predicted=float(np.average(p,weights=w)))
def raw_data():return {s:(pd.read_parquet(P/f'{s}_meta.parquet'),np.load(P/f'{s}_X.npy')) for s in ['mimic','eicu']}
def training_indices(d):
 n=json.loads((A/'design.json').read_text())['train_n_each_source'];ix=np.flatnonzero(d.split=='train');rank=np.array([h(v,'v10-training-size') for v in d.patient_id.iloc[ix]])
 return ix[np.argsort(rank)[:n]]
def transform(x,d,prep):
 med=np.array(prep['median']);mean=np.array(prep['mean']);std=np.array(prep['std'])
 mask=np.isnan(x).astype('float32');val=np.clip((np.where(np.isnan(x),med,x)-mean)/std,-10,10).astype('float32')
 seq=np.concatenate([val,mask],axis=2)
 static=np.stack([(d.age.to_numpy()-60)/20,(d.sex=='M').to_numpy(),(~d.sex.isin(['M','F'])).to_numpy(),np.repeat(24/72,len(d))],axis=1).astype('float32')
 return np.concatenate([seq.reshape(len(x),-1),static],axis=1)
def prepared(source,data):
 d,x=data[source];tr=training_indices(d);med=np.nanmedian(x[tr],axis=(0,1));med=np.nan_to_num(med);v=np.where(np.isnan(x[tr]),med,x[tr]);prep={'median':med.tolist(),'mean':v.mean((0,1)).tolist(),'std':np.maximum(v.std((0,1)),1e-3).tolist()}
 return {s:transform(xx,dd,prep) for s,(dd,xx) in data.items()},prep,tr
class Transformer(nn.Module):
 def __init__(self,d=16,layers=1):
  super().__init__();self.embed=nn.Linear(12,d);self.position=nn.Parameter(torch.zeros(1,6,d));self.static=nn.Linear(4,d)
  self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,d*2,.1,batch_first=True,activation='gelu'),layers,enable_nested_tensor=False);self.out=nn.Linear(d,1)
 def forward(self,x):return self.out(self.encoder(self.embed(x[:,:72].reshape(-1,6,12))+self.position).mean(1)+self.static(x[:,72:])).squeeze(1)
def predict(est,x,f):
 if f!='transformer':return est.predict_proba(x)[:,1]
 est.eval();out=[]
 with torch.no_grad():
  for j in range(0,len(x),4096):out.append(torch.sigmoid(est(torch.from_numpy(x[j:j+4096]))).numpy())
 return np.concatenate(out)
def fit(f,c,seed,x,y):
 np.random.seed(seed);torch.manual_seed(seed)
 if f=='logistic':m=LogisticRegression(C=c['C'],max_iter=1000,random_state=seed)
 elif f=='hist_boost':m=HistGradientBoostingClassifier(max_iter=200,learning_rate=c['lr'],max_leaf_nodes=c['leaves'],l2_regularization=1,min_samples_leaf=30,early_stopping=False,random_state=seed)
 elif f=='extra_trees':m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=c['leaf'],max_features=c['features'],n_jobs=4,random_state=seed)
 else:
  m=Transformer(c['d'],c['layers']);m.out.bias.data.fill_(float(logit(y.mean())))
  opt=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=.001);loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y.astype('float32'))),batch_size=1024,shuffle=True)
  for ep in range(30):
   m.train()
   for xb,yb in loader:
    opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(m(xb),yb);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  return m
 m.fit(x,y);return m
def calibrate(p,y):return LogisticRegression(C=1e3,max_iter=500).fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),y)
def apply_cal(c,p):return c.predict_proba(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1))[:,1]
def main():
 data=raw_data()
 for source in ['mimic','eicu']:
  xx,prep,tr=prepared(source,data);d=data[source][0];y=d.y.to_numpy();tu=np.flatnonzero(d.split=='tune');ca=np.flatnonzero(d.split=='calibrate')
  (P/f'{source}_preprocessing.json').write_text(json.dumps(prep,indent=2))
  for f in FAMILIES:
   chosenpath=A/f'{source}_{f}_chosen.json'
   if not chosenpath.exists():
    records=[];best=None;score=(-1,-1)
    for j,c in enumerate(CONFIGS[f]):
     start=time.perf_counter();log(f'{source} {f} candidate{j+1}/8');est=fit(f,c,42,xx[source][tr],y[tr]);mt=metrics(y[tu],predict(est,xx[source][tu],f))
     records.append(dict(source=source,model=f,candidate=j,params=json.dumps(c),train_n=len(tr),seconds=time.perf_counter()-start,**mt));sc=(mt['average_precision'],mt['auroc'])
     if sc>score:best=est;score=sc;bestc=c
     pd.DataFrame(records).to_csv(A/f'{source}_{f}_tuning.csv',index=False)
    joblib.dump(best,P/f'{source}_{f}_seed42_model.joblib');chosenpath.write_text(json.dumps({'params':bestc,'tune_ap':score[0],'tune_auc':score[1],'train_n':len(tr)},indent=2));del best,est;gc.collect()
   chosen=json.loads(chosenpath.read_text())
   for seed in [42,117,2026]:
    out=P/f'{source}_{f}_{seed}.npz'
    if out.exists():continue
    est=joblib.load(P/f'{source}_{f}_seed42_model.joblib') if seed==42 else fit(f,chosen['params'],seed,xx[source][tr],y[tr])
    raw={s:predict(est,x,f) for s,x in xx.items()};cal=calibrate(raw[source][ca],y[ca]);pred={s:apply_cal(cal,p) for s,p in raw.items()}
    np.savez_compressed(out,**pred,**{s+'_raw':p for s,p in raw.items()});joblib.dump({'model':est,'calibrator':cal,'prep':prep},P/f'{source}_{f}_{seed}_system.joblib')
    rows=[]
    for target,(dd,_) in data.items():
     te=np.flatnonzero(dd.split=='test');rows.append(dict(train_source=source,evaluation=target,model=f,seed=seed,**metrics(dd.y.to_numpy()[te],pred[target][te])))
    pd.DataFrame(rows).to_csv(A/f'{source}_{f}_{seed}_metrics.csv',index=False);log(f'finished {source} {f} seed{seed}');del est;gc.collect()
  tuning={f:json.loads((A/f'{source}_{f}_chosen.json').read_text()) for f in FAMILIES};selected=max(FAMILIES,key=lambda f:(tuning[f]['tune_ap'],tuning[f]['tune_auc']))
  (A/f'{source}_selected.json').write_text(json.dumps({'model':selected,'selection':'Source tuning average precision, AUROC tie break; primary seed42'},indent=2))
if __name__=='__main__':main()
