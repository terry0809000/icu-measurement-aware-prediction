"""Repeat all shuffled-label controls, retaining every permutation."""
from models import *
def main():
 data=raw_data();rows=pd.read_csv(A/'controls.csv');rows=rows[rows.analysis=='age_sex_only'].to_dict('records')
 for source in ['mimic','eicu']:
  xx,prep,tr=prepared(source,data);y=data[source][0].y.to_numpy()
  for k in range(20):
   seed=1942+k;yp=np.random.default_rng(seed).permutation(y[tr]);est=fit('hist_boost',{'leaves':15,'lr':.03},42,xx[source][tr],yp)
   for ev,(d,x) in data.items():
    te=np.flatnonzero(d.split=='test');p=predict(est,xx[ev][te],'hist_boost');rows.append(dict(train_source=source,evaluation=ev,analysis='shuffled_training_labels',permutation_seed=seed,**metrics(d.y.to_numpy()[te],p)))
   log(f'negative control {source} permutation{k+1}/20')
 pd.DataFrame(rows).to_csv(A/'controls.csv',index=False)
 d=pd.DataFrame(rows);d=d[d.analysis=='shuffled_training_labels'];d.groupby(['train_source','evaluation']).auroc.agg(['count','mean','std','min','max']).reset_index().to_csv(A/'negative_control_summary.csv',index=False)
if __name__=='__main__':main()
