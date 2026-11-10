"""Plot manuscript figures from the accompanying saved experiment results."""
from pathlib import Path
import argparse,shutil
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
OUT=ROOT/'figures'
plt.rcParams.update({'font.size':9,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22})
STYLES={'function':('#006DAA','o','Function'),'fairness':('#CC7914','s','Fairness'),'rate':('#26845C','^','Rate'),'power':('#9467BD','v','Power'),'gnn':('#CC4C54','D','GNN'),'random':('#707070','x','Random RIS'),'no_ris':('#303030','+','No RIS'),'ao_quantized':('#C3750E','s','Component AO (2-bit)'),'ao_continuous':('#288962','^','Component AO (continuous)'),'perfect_ao_continuous':('#8856A7','D','Perfect-CSI AO (continuous)')}
KINDS=['function','fairness','rate','power']

def interval(values):
    x=np.asarray(values,float)
    return x.mean(),t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))

def save(fig,number):
    for ext in ('pdf','png'):fig.savefig(OUT/f'figure_{number:02}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def phy(number):
    records={}
    for seed in range(721,731):
        with np.load(DATA/f'phy/{seed}.npz') as z:
            for key in z.files:
                assert z[key].shape==(1000,) and np.isfinite(z[key]).all()
                records.setdefault(key,[]).append(z[key].mean())
    sweep,grid,xlabel={2:('snr',[0,10,20,30,40],'SNR (dB)'),3:('L',[2,4,8,16,32],'Separately trained codebook size L'),4:('rho',[0,.01,.05,.1,.2],r'Normalized CSI uncertainty $\epsilon$')}[number]
    fig,axes=plt.subplots(1,1 if number==3 else 2,figsize=(3.65,2.65) if number==3 else (7.2,3.1),layout='constrained')
    axes=np.atleast_1d(axes)
    series=[KINDS] if number==3 else [KINDS+['gnn','random','no_ris'],['function','ao_quantized','ao_continuous','perfect_ao_continuous']]
    for ax,methods in zip(axes,series):
        for method in methods:
            rows=[interval(records[f'{sweep}|{v}|{method}|conditional_nmse']) for v in grid]
            color,marker,label=STYLES[method]
            ax.errorbar(grid,[r[0] for r in rows],yerr=[r[1] for r in rows],color=color,marker=marker,label=label,ms=4,mfc='white',lw=1.15,capsize=2)
        ax.set(xlabel=xlabel,ylabel='Conditional aggregation NMSE',yscale='log')
        ax.set_xticks(grid);ax.legend(ncol=2 if len(methods)>4 else 1)
    if number==3:axes[0].set_xscale('log',base=2);axes[0].set_xticks(grid,[str(x) for x in grid])
    else:
        axes[0].set_title('(a) Books and learned baseline');axes[1].set_title('(b) Local AO references')
        if number==4:
            for ax in axes:ax.set_xticks([0,.05,.1,.15,.2])
    save(fig,number)

def fl():
    h=pd.read_csv(DATA/'fl_rounds.csv');cost=pd.read_csv(DATA/'fl_target_cost.csv')
    methods=['function','fairness','rate','ao_qDFT65']
    colors=['#b32624','#247ba0','#e19a29','#33815c','#333333'];labels=['Function','Fairness','Rate','Component AO','Ideal']
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,tau in enumerate([2000,50000]):
        for method,color,label in zip(methods+['ideal'],colors,labels):
            g=h[(h.tau==tau)&(h.method==method)].dropna(subset=['accuracy']).groupby(['environment_seed','round']).accuracy.mean().groupby('round').agg(['mean','std'])
            axs[row,0].plot(g.index,g['mean']*100,color=color,label=label,ls='--' if method in ['ideal','ao_qDFT65'] else '-')
            if method in methods[:3]:axs[row,0].fill_between(g.index,(g['mean']-g['std'])*100,(g['mean']+g['std'])*100,color=color,alpha=.08)
        axs[row,0].set(title=f'({chr(97+2*row)}) Coherence {tau:,} uses',xlabel='Communication round',ylabel='Test accuracy (%)');axs[row,0].legend(ncol=2,loc='lower right')
        for i,threshold in enumerate([.7,.75,.8]):
            for j,(method,color,label) in enumerate(zip(methods,colors,labels)):
                q=cost[(cost.tau==tau)&(cost.method==method)&(cost.rule=='first')&np.isclose(cost.threshold,threshold)]
                assert len(q)==30, 'Expected all 30 trajectories for each target'
                reached=q.reached.astype(str).str.lower().eq('true')
                if not reached.all():
                    axs[row,1].text(i+(j-1.5)*.19,.08,f'{int(reached.sum())}/30',rotation=90,ha='center',fontsize=7)
                    continue
                values=q.groupby('environment_seed').channel_uses.mean().to_numpy();assert len(values)==10
                mean,half=interval(values);x=i+(j-1.5)*.19
                axs[row,1].bar(x,mean/1e6,.18,color=color,label=label if i==0 else None,yerr=half/1e6,error_kw={'capsize':2,'elinewidth':.8})
                axs[row,1].scatter(x+np.linspace(-.06,.06,10),values/1e6,s=7,c='black',alpha=.55,zorder=3)
        axs[row,1].set(title=f'({chr(98+2*row)}) First observed crossing',xlabel='Target accuracy',ylabel='Uplink uses (million)',ylim=(0,None));axs[row,1].set_xticks(range(3),['70%','75%','80%']);axs[row,1].legend(ncol=2)
    save(fig,5)

def bound():
    d=pd.read_csv(DATA/'bounds.csv');assert len(d)==240
    grid=[2,4,8,16];fig,axs=plt.subplots(1,2,figsize=(10,3.65),layout='constrained')
    for key,label,color,marker in [('old_upper','Original bound','#b53b3b','s'),('tx_upper','One transmit update','#2274a5','o'),('txrx_upper','Transmit + receive update','#198560','^')]:
        axs[0].plot(grid,[d[d.L==v][key].mean() for v in grid],color=color,marker=marker,label=label)
        axs[1].plot(grid,[(d[d.L==v][key]/d[d.L==v].gap_lower).median() for v in grid],color=color,marker=marker,label=label)
    axs[0].plot(grid,[d[d.L==v].gap_lower.mean() for v in grid],'k--',label='Ideal gap (numerical reference)')
    axs[0].set(ylabel='Mean aggregation-MSE loss / bound',title='(a) Same 60 states at each codebook size')
    axs[1].set(ylabel='Median bound / ideal-gap ratio',title='(b) Tightening without fitted coefficients');axs[1].axhline(1,color='.45',ls=':')
    for ax in axs:
        ax.set(yscale='log',xlabel='Codebook size, L');ax.set_xscale('log',base=2);ax.set_xticks(grid,grid)
    axs[0].legend();save(fig,6)

def sensitivity():
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    methods=['function_3','fairness_3','rate_3','power_3','random','ao'];colors=['#0072B2','#E69F00','#009E73','#CC79A7','#777777','#222222'];markers=['o','s','^','D','v','x'];labels=['Function','Fairness','Rate','Power','Random RIS','Component AO']
    for ax,(folder,prefix,grid,title,xlabel) in zip(axs.flat,[('ris','m',[32,64,128],'(a) RIS size','RIS elements M'),('receive','n',[2,4,8],'(b) Receive array','Receive antennas'),('phase','b',[1,2,3],'(c) Phase resolution','Phase resolution (bits)'),('size','n',[250,500,1000,2000],'(d) Training size','Training states')]):
        for method,color,marker,label in zip(methods,colors,markers,labels):
            values=[]
            for v in grid:
                a=[]
                for seed in range(1201,1211):
                    with np.load(DATA/f'{folder}/{seed}_{prefix}{v}.npz') as z:a.append(z[method].mean())
                values.append(a)
            stats=[interval(a) for a in values];ax.errorbar(grid,[x[0] for x in stats],yerr=[x[1] for x in stats],marker=marker,color=color,label=label,ls='--' if method in ['random','ao'] else '-',capsize=2,ms=4)
            width=min(np.diff(grid))*.025
            for v,a in zip(grid,values):ax.scatter(v+np.linspace(-width,width,10),a,s=6,color=color,alpha=.25)
        ax.set(title=title,xlabel=xlabel,ylabel='Conditional aggregation NMSE',ylim=(0,None));ax.set_xticks(grid)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6);save(fig,7)

def main():
    global DATA, OUT
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['all']+[str(i) for i in range(1,8)],default='all');parser.add_argument('--data-dir',type=Path,default=DATA);parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();DATA=args.data_dir;OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    for n in range(1,8) if args.figure=='all' else [int(args.figure)]:
        if n==1:shutil.copy2(ROOT/'reference_figures/figure_01.pdf',OUT/'figure_01.pdf')
        elif n in [2,3,4]:phy(n)
        else:{5:fl,6:bound,7:sensitivity}[n]()
        print(f'Figure {n}: {OUT / ("figure_%02d.pdf" % n)}')

if __name__=='__main__':main()
