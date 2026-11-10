from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parent
rows=[]
for e in [721,722,723]:
 for off in [0,100000,200000]:
  seed=e+off
  for n,suffix in [(800,'_fullbatch'),(3200,'')]:
   q=json.loads((R/'runs'/f'{e}_{n}_{seed}{suffix}'/'report.json').read_text())
   assert q['samples_seen']==166400 and q['stop_step']==2600
   rows.append({k:v for k,v in q.items() if k!='history'})
df=pd.DataFrame(rows);df.to_csv(R/'matched_exposure_runs.csv',index=False)
p=df.pivot(index=['environment','initialization_seed'],columns='n_train',values='test_nmse');p['difference']=p[3200]-p[800];p.to_csv(R/'matched_exposure_pairs.csv')
means=df.groupby(['environment','n_train']).test_nmse.mean().unstack();delta=means[3200]-means[800]
for e in [721,722,723]:
 for off in [0,100000,200000]:
  q=df[(df.environment==e)&(df.initialization_seed==e+off)];assert q.initial_weights_hash.nunique()==1
summary={'matched_exposure_training_runs':len(df),'mean_test_nmse':{str(n):float(means[n].mean()) for n in [800,3200]},'relative_reduction_percent':float(100*(1-means[3200].mean()/means[800].mean())),'paired_difference_mean':float(delta.mean()),'environment_t95_ci':[float(delta.mean()-t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3)),float(delta.mean()+t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3))],'environment_wins':int((delta<0).sum()),'paired_initialization_wins':int((p.difference<0).sum()),'environment_means':means.reset_index().to_dict('records'),'seed_std':df.groupby(['environment','n_train']).test_nmse.std().reset_index().to_dict('records'),'status':'secondary matched-exposure control','fairness':'same2600optimizer steps and166400 processed samples, fixed64batch; different unique data count'}
(R/'control_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
