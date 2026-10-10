from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MaxNLocator
ROOT=Path(__file__).resolve().parent;A=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
MODELS=['logistic','hist_boost','extra_trees','transformer'];NAMES=['Logistic','Boosting','Extra trees','Transformer'];C=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
def save(fig,name):fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight',dpi=300);plt.close(fig)
def panel(ax,s):ax.text(-.13,1.08,s,transform=ax.transAxes,fontweight='bold',fontsize=12)
def main():
 missing=pd.read_csv(A/'missingness.csv');bal=pd.read_csv(A/'balance.csv');met=pd.read_csv(A/'metrics.csv');native=met[met.regime=='native'];conf=pd.read_csv(A/'conformal.csv')
 fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 features=['hr','rr','spo2','sbp','dbp','map']
 for j,s in enumerate(['mimic','eicu']):
  d=missing[(missing.source==s)&(missing.split=='test')].set_index('feature').loc[features];axs[0].barh(np.arange(6)+(j-.5)*.3,d.missing_fraction,.28,color=C[j],label=['MIMIC','eICU'][j]);den=pd.read_csv(A/f'{s}_density.csv').set_index('feature').loc[features];axs[1].barh(np.arange(6)+(j-.5)*.3,den['mean'],.28,color=C[j])
 for ax,l in zip(axs,'AB'):ax.set_yticks(range(6),['Heart rate','Respiratory rate','SpO2','Cuff SBP','Cuff DBP','Cuff MAP']);ax.invert_yaxis();panel(ax,l)
 axs[0].xaxis.set_major_formatter(PercentFormatter(1));axs[0].set_xlabel('Missing hourly inputs in test patients');axs[0].legend();axs[1].set_xlabel('Observations per populated hourly bin');save(fig,'figure1_measurements')
 ints=pd.concat([pd.read_csv(p) for p in A.glob('intervals_*_patient_id.csv') if 'overlap' not in p.name])
 fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for i,source in enumerate(['mimic','eicu']):
  for j,metric in enumerate(['auroc','average_precision']):
   ax=axs[i,j]
   for k,ev in enumerate(['mimic','eicu']):
    d=ints[(ints.analysis==source+'_to_'+ev)&(ints.metric==metric)].set_index('model').loc[MODELS];ax.errorbar(d.estimate,np.arange(4)+(k-.5)*.15,xerr=[d.estimate-d.lower,d.upper-d.estimate],fmt=['o','s'][k],color=C[k],capsize=3,label=['MIMIC test','eICU hospital test'][k])
   ax.set_yticks(range(4),NAMES);ax.invert_yaxis();ax.set_xlabel('AUROC' if j==0 else 'Average precision');ax.set_title('Trained on '+source.upper());ax.grid(axis='x',alpha=.2);panel(ax,chr(65+2*i+j))
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.08),ncol=2);save(fig,'figure2_reciprocal')
 gap=pd.read_csv(A/'auc_gaps.csv');fig,axs=plt.subplots(1,2,figsize=(9.2,4),layout='constrained')
 for i,s in enumerate(['mimic','eicu']):
  d=gap[gap.train_source==s].set_index('model').loc[MODELS];axs[0].errorbar(d.external_minus_source_auc,np.arange(4)+(i-.5)*.18,xerr=[d.external_minus_source_auc-d.lower,d.upper-d.external_minus_source_auc],fmt=['o','s'][i],color=C[i],capsize=3,label=['MIMIC to eICU','eICU to MIMIC'][i])
 axs[0].axvline(0,color='gray',ls='--');axs[0].set_yticks(range(4),NAMES);axs[0].invert_yaxis();axs[0].set_xlabel('External minus source-test AUROC');axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2);panel(axs[0],'A')
 y=np.arange(len(bal));axs[1].scatter(abs(bal.smd_native),y,label='Native',color=C[0]);axs[1].scatter(abs(bal.smd_overlap),y,label='Overlap weighted',color=C[1]);axs[1].set_yticks(y,bal.feature.str.replace('_',' '));axs[1].invert_yaxis();axs[1].set_xlabel('Absolute standardised mean difference');axs[1].legend();panel(axs[1],'B');save(fig,'figure3_auc_gap')
 curves=pd.read_csv(A/'calibration_curves.csv');fig,axs=plt.subplots(2,2,figsize=(8.5,6.5),layout='constrained')
 for i,s in enumerate(['mimic','eicu']):
  for j,e in enumerate(['mimic','eicu']):
   ax=axs[i,j];d=curves[(curves.train_source==s)&(curves.evaluation==e)];lim=max(d.predicted.max(),d.observed.max())*1.05
   for f,name,c in zip(MODELS,NAMES,C):g=d[d.model==f];ax.plot(g.predicted,g.observed,'o-',label=name,color=c,markersize=3)
   ax.plot([0,lim],[0,lim],'--',color='gray');ax.set_xlim(0,lim);ax.set_ylim(0,lim);ax.set_xlabel('Mean predicted death risk');ax.set_ylabel('Observed ICU mortality');ax.set_title(s.upper()+' trained; '+e.upper()+' test');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));panel(ax,chr(65+2*i+j))
 axs[0,0].legend();save(fig,'figure4_calibration')
 fig,axs=plt.subplots(2,3,figsize=(10,6.5),layout='constrained')
 for i,s in enumerate(['mimic','eicu']):
  e='eicu' if s=='mimic' else 'mimic';f=json.loads((A/f'{s}_selected.json').read_text())['model'];spec=[(s,'marginal_failure_baseline'),(e,'marginal_failure_baseline'),(s,'class_conditional'),(e,'class_conditional'),(e,'local_class_conditional')]
  d=pd.DataFrame([conf[(conf.train_source==s)&(conf.evaluation==ev)&(conf.model==f)&(conf.method==m)].iloc[0] for ev,m in spec])
  for j,col in enumerate(['event_coverage','both_rate','empty_rate']):
   ax=axs[i,j];ax.bar(range(5),d[col],color=[C[0],C[1],C[0],C[1],C[2]],alpha=.85)
   if j==0:ax.errorbar(range(5),d[col],yerr=[d[col]-d.event_lower,d.event_upper-d[col]],fmt='none',color='black',capsize=3);ax.axhline(.9,color='gray',ls='--')
   ax.set_ylim(0,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xticks(range(5),['Source\nmarginal','Target\nmarginal','Source\nclasswise','Target\nclasswise','Target\nupdated'],rotation=35,ha='right');ax.set_title(['Death-class coverage','Two-label sets','Empty sets'][j]);panel(ax,chr(65+3*i+j))
   if j==0:ax.set_ylabel(s.upper()+'-selected '+NAMES[MODELS.index(f)])
 save(fig,'figure5_conformal')
 thin=pd.read_csv(A/'thinning.csv');fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for i,s in enumerate(['mimic','eicu']):
  for j,e in enumerate(['mimic','eicu']):
   ax=axs[i,j]
   for k,f in enumerate(MODELS):
    r=native[(native.train_source==s)&(native.evaluation==e)&(native.model==f)].iloc[0];d=thin[(thin.train_source==s)&(thin.evaluation==e)&(thin.model==f)];ax.plot([k-.12,k+.12],[r.auroc,d.auroc.mean()],'-o',color=C[k]);ax.errorbar(k+.12,d.auroc.mean(),yerr=[[max(0,d.auroc.mean()-d.auroc.min())],[max(0,d.auroc.max()-d.auroc.mean())]],color=C[k],capsize=3)
   ax.set_xticks(range(4),NAMES,rotation=20,ha='right');ax.set_ylabel('AUROC');ax.set_title(s.upper()+' trained; '+e.upper()+' test');panel(ax,chr(65+2*i+j))
 fig.supxlabel('Within each model: native inputs (left), symmetrically thinned inputs (right)',fontsize=9);save(fig,'figure6_measurement_stress')
 se=pd.read_csv(A/'sensitivities.csv');fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained');spec=['no_blood_pressure','without_explicit_masks','native_death_within24h','exit_death_within24h']
 for j,s in enumerate(['mimic','eicu']):
  for i,e in enumerate(['mimic','eicu']):
   d=se[(se.train_source==s)&(se.evaluation==e)].set_index('analysis').loc[spec];axs[j].plot(d.auroc,np.arange(4)+(i-.5)*.12,'o',color=C[i],label=e.upper())
  axs[j].set_yticks(range(4),['No BP','No explicit masks','Native 24h death','Exit-offset 24h death']);axs[j].invert_yaxis();axs[j].set_xlabel('AUROC (different tasks shown separately)');axs[j].set_title(s.upper()+' trained boosting');axs[j].legend();panel(axs[j],chr(65+j))
 save(fig,'figureS1_sensitivities')
 dec=pd.read_csv(A/'decision.csv');fig,axs=plt.subplots(2,2,figsize=(8.5,6),layout='constrained')
 for j,s in enumerate(['mimic','eicu']):
  e='eicu' if s=='mimic' else 'mimic';d=dec[(dec.train_source==s)&(dec.evaluation==e)]
  for f,name,c in zip(MODELS,NAMES,C):
   g=d[d.model==f];axs[0,j].plot(g.threshold,g.net_benefit,label=name,color=c);axs[1,j].plot(g.threshold,g.alerts_per100,color=c)
  axs[0,j].plot(g.threshold,g.treat_all,':',color='gray',label='Assess all');axs[0,j].axhline(0,color='black',ls='--');axs[0,j].set_ylim(-.03,.08);axs[0,j].set_ylabel('Net benefit');axs[0,j].set_title(s.upper()+' to '+e.upper());axs[1,j].set_ylabel('Assessments per 100 patients')
  for i in range(2):axs[i,j].set_xlabel('Illustrative risk threshold');axs[i,j].xaxis.set_major_formatter(PercentFormatter(1));panel(axs[i,j],chr(65+2*i+j))
 axs[0,0].legend();save(fig,'figureS2_decision')
 print('Generated eight figure pairs',flush=True)
if __name__=='__main__':main()
