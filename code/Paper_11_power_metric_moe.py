"""
MoE Cross-Expert Health Monitoring via Power Metric
=====================================================
Paper 11: "Cross-Expert Health Monitoring in Mixture-of-Experts
           via Stochastic Power Metrics"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies cross-expert P_i(t) to correct static router bias in MoE.

E_i(t) = q_i(t) / global_mean_quality(t)  [cross-expert reference]
score_i = (1-β)×router_i + β×normalize(P_i(t))

Note: stylized simulation. Validation requires real MoE routing traces.

Related: https://github.com/HauntedKernel/power-metric
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ALPHA=0.3; LAMBDA=0.5; EWMA_SPAN=3
N_EXPERTS=8; N_TOKENS=600; N_SEEDS=5
GENERAL_BIAS=0.28; NOISE_Q=0.02; NOISE_L=0.03
EXPERT_NAMES=['General','Math-A','Math-B','Code-A','Code-B','Lang-A','Lang-B','Reas-A']
TOKEN_TYPES=['Math','Code','Language','Reasoning']

def build_quality(seed=42):
    rng=np.random.default_rng(seed)
    s=lambda m: rng.uniform(0.87,0.92) if m else rng.uniform(0.53,0.57)
    g=lambda: rng.uniform(0.68,0.70)
    return np.array([
        [g(),g(),g(),g()],
        [s(True),s(False),s(False),s(False)],
        [s(True),s(False),s(False),s(False)],
        [s(False),s(True),s(False),s(False)],
        [s(False),s(True),s(False),s(False)],
        [s(False),s(False),s(True),s(False)],
        [s(False),s(False),s(True),s(False)],
        [s(False),s(False),s(False),s(True)],
    ])

def softmax(x):
    e=np.exp(x-x.max()); return e/e.sum()

def norm_pm(pm):
    mn,mx=pm.min(),pm.max()
    if mx==mn: return np.ones(len(pm))/len(pm)
    return (pm-mn)/(mx-mn)

def simulate_one(beta, quality, seed):
    rng=np.random.default_rng(seed)
    state=[{'er':None,'ew':0.0,'pw':0.5} for _ in range(N_EXPERTS)]
    logits=np.zeros((N_EXPERTS,4)); logits[0,:]=GENERAL_BIAS
    tc=np.zeros(N_EXPERTS); quals=[]
    for t in range(N_TOKENS):
        tt=t//150
        sp=softmax(logits[:,tt]+rng.normal(0,NOISE_L,N_EXPERTS))
        pm_raw=np.array([s['pw'] for s in state])
        bl=(1-beta)*sp+beta*norm_pm(pm_raw); bl/=bl.sum()
        top2=np.argsort(bl)[-2:]
        gm=np.mean(quality[:,tt])
        for e in top2:
            tc[e]+=1
            q=quality[e,tt]+rng.normal(0,NOISE_Q); quals.append(q)
            sig=q/gm if gm>1e-6 else 1.0
            st=state[e]
            if st['er'] is None: eff=1.0; st['er']=max(sig,1e-6)
            else: eff=sig/st['er']; st['er']=(1-ALPHA)*st['er']+ALPHA*sig
            win=1.0 if eff>1.0 else 0.0
            a=2.0/(EWMA_SPAN+1); st['ew']=a*win+(1-a)*st['ew']
            inst=eff*st['ew']
            st['pw']=np.exp(-LAMBDA)*st['pw']+(1-np.exp(-LAMBDA))*inst
    n=N_EXPERTS; sv=np.sort(tc/tc.sum())
    gini=(2*np.sum(np.arange(1,n+1)*sv)-(n+1))/n
    return float(np.mean(quals)), gini, tc

def run_analysis(betas=None, qseed=42):
    if betas is None: betas=[0.0,0.1,0.2,0.3,0.4,0.5]
    quality=build_quality(qseed); results={}
    for beta in betas:
        qs=[]; gs=[]; tcs=[]
        for s in range(N_SEEDS):
            q,g,tc=simulate_one(beta,quality,s)
            qs.append(q); gs.append(g); tcs.append(tc)
        results[beta]=dict(quality=np.mean(qs),gini=np.mean(gs),
                           token_counts=np.mean(tcs,axis=0))
    return results, quality

def print_summary(results):
    bq=results[0.0]['quality']; bg=results[0.0]['gini']
    bgen=results[0.0]['token_counts'][0]
    print(f"{'B':>5} {'Quality':>9} {'DQ':>8} {'Gini':>7} {'DG':>8} {'General':>8}")
    print("-"*50)
    for b,r in results.items():
        dq=(r['quality']-bq)/bq*100; dg=(r['gini']-bg)/bg*100
        print(f"{b:>5.1f} {r['quality']:>9.4f} {dq:>+7.2f}% "
              f"{r['gini']:>7.4f} {dg:>+7.2f}% {r['token_counts'][0]:>8.0f}")

def plot_results(results, quality, save_path=None):
    fig=plt.figure(figsize=(14,10),facecolor='#050810')
    gs=gridspec.GridSpec(3,2,figure=fig,hspace=0.5,wspace=0.38)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    BG='#050810'; PAN='#0d1117'; GR='#888888'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    betas=sorted(results.keys())
    bq=results[0.0]['quality']; bg=results[0.0]['gini']
    dq=[(results[b]['quality']-bq)/bq*100 for b in betas]
    dg=[(results[b]['gini']-bg)/bg*100 for b in betas]
    generals=[results[b]['token_counts'][0] for b in betas]

    ax1=fig.add_subplot(gs[0,0]); style(ax1)
    ax1.plot(betas,dq,color=CG,lw=2,marker='o',ms=7)
    ax1.axhline(0,color='white',lw=1,ls='--',alpha=0.4)
    for x,y in zip(betas,dq):
        if x>0: ax1.annotate(f'{y:+.1f}%',(x,y),textcoords='offset points',
                               xytext=(0,8),ha='center',color=CG,fontsize=8)
    ax1.set_title('Quality Improvement vs β',color=C,fontsize=11)
    ax1.set_xlabel('β',color=GR); ax1.set_ylabel('Δ Quality (%)',color=GR)

    ax2=fig.add_subplot(gs[0,1]); style(ax2)
    ax2.plot(betas,dg,color=CA,lw=2,marker='o',ms=7)
    ax2.axhline(0,color='white',lw=1,ls='--',alpha=0.4)
    ax2.set_title('Gini Change vs β\n(negative=more balanced)',color=C,fontsize=10)
    ax2.set_xlabel('β',color=GR); ax2.set_ylabel('Δ Gini (%)',color=GR)

    ax3=fig.add_subplot(gs[1,0]); style(ax3)
    ax3.plot(betas,generals,color=CB,lw=2,marker='o',ms=7)
    ax3.axhline(600,color=GR,lw=1,ls='--',label='Max (600)')
    ax3.set_title('General Expert Usage vs β',color=C,fontsize=11)
    ax3.set_xlabel('β',color=GR); ax3.set_ylabel('Tokens',color=GR)
    ax3.set_ylim(0,650)

    ax4=fig.add_subplot(gs[1,1]); style(ax4)
    tc0=results[0.0]['token_counts']; tc2=results[0.2]['token_counts']
    xp=np.arange(N_EXPERTS)
    ax4.bar(xp-0.2,tc0,0.35,color=GR,alpha=0.7,label='Static β=0')
    ax4.bar(xp+0.2,tc2,0.35,color=C,alpha=0.85,label='PM β=0.2')
    ax4.set_xticks(xp); ax4.set_xticklabels([e[:7] for e in EXPERT_NAMES],
                                              fontsize=7,rotation=20)
    ax4.set_title('Expert Utilization: Static vs PM (β=0.2)',color=C,fontsize=10)
    ax4.set_ylabel('Tokens',color=GR)
    ax4.legend(fontsize=8,labelcolor='white',facecolor=PAN,edgecolor='#333')

    ax5=fig.add_subplot(gs[2,:]); style(ax5)
    ax5t=ax5.twinx(); ax5t.set_facecolor(PAN)
    ax5.plot(betas,dq,color=CG,lw=2,marker='o',ms=6,label='Δ Quality (%)')
    ax5t.plot(betas,dg,color=CA,lw=2,marker='s',ms=6,ls='--',label='Δ Gini (%)')
    ax5.axhline(0,color='white',lw=0.5,ls=':',alpha=0.3)
    ax5t.tick_params(colors='#666666')
    ax5.set_title('Quality vs Load Balance Tradeoff by β',color=C,fontsize=11)
    ax5.set_xlabel('β',color=GR)
    ax5.set_ylabel('Δ Quality (%)',color=CG)
    ax5t.set_ylabel('Δ Gini (%)',color=CA)
    ax5.legend(loc='upper left',fontsize=8,labelcolor='white',
               facecolor=PAN,edgecolor='#333')
    ax5t.legend(loc='upper right',fontsize=8,labelcolor='white',
                facecolor=PAN,edgecolor='#333')

    fig.suptitle('MoE Cross-Expert Health Monitoring — P_i(t) = E_i(t)×W_i(t)\n'
                 '8 experts · 600 tokens · General bias +0.28 · '
                 'Stylized simulation · Validation requires real MoE traces',
                 color=C,fontsize=10,fontweight='bold',y=0.99)
    plt.figtext(0.5,0.01,
                f'α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN} · '
                'E_i(t) = q_i/global_mean · Cantrell (2026) · Paper 11 · '
                'github.com/HauntedKernel/power-metric',
                ha='center',color='#444',fontsize=8)

    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight',facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig

if __name__=='__main__':
    print("MoE Cross-Expert Health Monitoring — Paper 11\n")
    results,quality=run_analysis()
    print_summary(results)
    print("\nGenerating charts...")
    plot_results(results,quality,
                 save_path='/mnt/user-data/outputs/paper11_simulation.png')
    print("Done.")
