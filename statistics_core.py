import numpy as np,pandas as pd
from scipy.special import logit,expit
from scipy.optimize import minimize
def calibration_stats(y,p,clusters=None):
 z=logit(np.clip(p,1e-7,1-1e-7));x=np.stack([np.ones(len(y)),z],axis=1)
 def fun(b):
  v=x@b;return np.mean(np.logaddexp(0,v)-y*v),x.T@(expit(v)-y)/len(y)
 opt=minimize(fun,[0.,1.],jac=True,method='BFGS')
 out={'calibration_intercept':float(opt.x[0]),'calibration_slope':float(opt.x[1]),'calibration_fit_converged':bool(opt.success)}
 if clusters is not None:
  mu=expit(x@opt.x);bread=np.linalg.pinv(x.T@(x*(mu*(1-mu))[:,None]))
  codes,groups=pd.factorize(clusters);score=x*(y-mu)[:,None]
  summed=np.stack([np.bincount(codes,weights=score[:,j],minlength=len(groups)) for j in range(2)],axis=1)
  cov=bread@(summed.T@summed)@bread*len(groups)/max(1,len(groups)-1);se=np.sqrt(np.maximum(np.diag(cov),0))
  for j,name in enumerate(['calibration_intercept','calibration_slope']):out[name+'_lower']=float(opt.x[j]-1.96*se[j]);out[name+'_upper']=float(opt.x[j]+1.96*se[j])
 return out

class FastMetric:
 def __init__(self,y,p):
  self.y=np.asarray(y,float);self.p=np.asarray(p,float)
  self.order=np.argsort(-self.p,kind='stable');ps=self.p[self.order]
  self.starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1]
  self.ys=self.y[self.order];self.loss=(self.y-self.p)**2
 def __call__(self,w):
  wp=w[self.order];pos=np.add.reduceat(wp*self.ys,self.starts);neg=np.add.reduceat(wp*(1-self.ys),self.starts)
  tp=np.cumsum(pos);fp=np.cumsum(neg);P=tp[-1];N=fp[-1]
  if P<=0 or N<=0:return [np.nan]*3
  ap=np.sum(np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)*pos)/P
  auc=np.sum(pos*(N-fp+.5*neg))/(P*N)
  return [auc,ap,np.dot(w,self.loss)/w.sum()]

def conformal_q(y,p,alpha=.1,cls=None):
 score=np.where(y==1,1-p,p)
 if cls is not None:score=score[y==cls]
 k=int(np.ceil((len(score)+1)*(1-alpha)))
 return 1. if k>len(score) else float(np.partition(score,k-1)[k-1])

def conformal_metrics(y,p,q0,q1):
 inc0=p<=q0;inc1=1-p<=q1;covered=np.where(y==1,inc1,inc0);size=inc0.astype(int)+inc1.astype(int)
 return {'n':len(y),'events':int(y.sum()),'coverage':float(covered.mean()),'event_coverage':float(covered[y==1].mean()) if y.sum() else np.nan,'nonevent_coverage':float(covered[y==0].mean()),'empty_rate':float((size==0).mean()),'both_rate':float((size==2).mean()),'mean_set_size':float(size.mean()),'q0':q0,'q1':q1}
