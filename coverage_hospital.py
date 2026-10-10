import numpy as np,pandas as pd
from models import *
from statistics_core import conformal_q
def main():
 data=raw_data();ed=data['eicu'][0];te=np.flatnonzero(ed.split=='test');y=ed.y.to_numpy()[te];codes,sites=pd.factorize(ed.site_id.iloc[te]);n=len(sites);rows=[]
 for source in ['mimic','eicu']:
  sd=data[source][0];ci=np.flatnonzero(sd.split=='conformal');cy=sd.y.to_numpy()[ci]
  for f in FAMILIES:
   z=np.load(P/f'{source}_{f}_42.npz');cp=z[source][ci];p=z['eicu'][te]
   for method in ['marginal_failure_baseline','class_conditional']:
    q=conformal_q(cy,cp);q0=q if method.startswith('marginal') else conformal_q(cy,cp,cls=0);q1=q if method.startswith('marginal') else conformal_q(cy,cp,cls=1)
    a=p<=q0;b=1-p<=q1;covered=np.where(y==1,b,a);vals=np.stack([np.ones(len(y)),y,covered,y*covered,a&b,~(a|b)],axis=1);tot=np.stack([np.bincount(codes,weights=vals[:,k],minlength=n) for k in range(6)],axis=1)
    rng=np.random.default_rng(20342);counts=rng.multinomial(n,np.repeat(1/n,n),size=1000);draw=counts@tot
    for metric,num,den in [('coverage',2,0),('event_coverage',3,1),('both_rate',4,0),('empty_rate',5,0)]:
     v=draw[:,num]/draw[:,den];lo,hi=np.quantile(v,[.025,.975]);rows.append(dict(train_source=source,evaluation='eicu',model=f,method=method,metric=metric,lower=lo,upper=hi,hospitals=n,replicates=1000))
 pd.DataFrame(rows).to_csv(A/'conformal_hospital_intervals.csv',index=False)
if __name__=='__main__':main()
