#!/usr/bin/env python3
"""Static supplementary figure for the matched null and finite-relay scan."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Paper/tables/supplement_audit.json'
OUT = ROOT / 'Paper/figs/supp/SuppFig_repair_controls'


def main():
    stats = json.loads(SOURCE.read_text())['tables']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(7.0,2.65),gridspec_kw={'width_ratios':[1,1.55]})
    totals=[]
    for variant in ['random-waypoint','MORPHICA']:
        rows=[r for r in stats['null'] if r['variant']==variant]
        n=sum(r['n'] for r in rows);events=sum(r['n']-r['failures'] for r in rows)
        totals.append({'variant':variant,'n':n,'successes':events})
    ax=axes[0]
    ax.bar([0,1],[r['successes']/r['n'] for r in totals],color=['#909090','#007C6C'],width=.55)
    for i,r in enumerate(totals):
        ax.text(i,r['successes']/r['n']+.035,f"{r['successes']}/{r['n']}",ha='center')
    ax.set_ylim(0,1.15);ax.set_xticks([0,1],['Random\nwaypoint','MORPHICA'])
    ax.set_ylabel('Observed recovery fraction');ax.set_title('a  Matched movement null',loc='left',fontweight='bold')
    ax.text(.04,.83,'N=1000\ncutoff 1799',transform=ax.transAxes,ha='left',fontsize=7)
    ax=axes[1];budgets=['1','2','4','8','16','infinite'];x=np.arange(len(budgets))
    for d,color in [(50,'#3579AD'),(100,'#DA8C24'),(150,'#AE555C')]:
        rows=[next(r for r in stats['relay'] if r['D']==d and r['variant']==b) for b in budgets]
        mean=np.array([r['rmst'] for r in rows]);low=np.array([r['rmst_ci'][0] for r in rows]);high=np.array([r['rmst_ci'][1] for r in rows])
        ax.plot(x,mean/799,'o-',color=color,label=f'{d/2:.0f}% node loss',lw=1.2,ms=3)
        ax.fill_between(x,low/799,high/799,color=color,alpha=.13,lw=0)
    ax.set_xticks(x,['1','2','4','8','16','Exact']);ax.set_xlabel('Relay rounds per movement update')
    ax.set_ylabel('RMST / recording cutoff');ax.set_ylim(bottom=0)
    ax.set_title('b  Finite domain signalling',loc='left',fontweight='bold')
    ax.legend(frameon=False,fontsize=7);ax.text(.03,.06,'N=200',transform=ax.transAxes,ha='left',fontsize=7)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=3)
    fig.tight_layout(pad=1.1,w_pad=2)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT.with_suffix('.pdf'))
    fig.savefig(OUT.with_suffix('.png'),dpi=300)
    OUT.with_suffix('.stats.json').write_text(json.dumps({
        'source':str(SOURCE.relative_to(ROOT)),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'null':totals,'relay':stats['relay'],
        'intervals':'within-condition 2000-resample percentile bootstrap; null bars descriptive'},indent=2)+'\n')
    print(OUT.with_suffix('.pdf'))


if __name__=='__main__':main()
