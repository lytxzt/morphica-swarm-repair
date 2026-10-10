#!/usr/bin/env python3
"""Build auditable SI tables from frozen logs and the supplemental reruns.

Tables are inserted inline between marked blocks in sn-article.tex, preserving
the journal template's single-TeX-source convention. Historical logs are read
only. The JSON audit records source hashes, case membership and estimands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from run_manuscript_evidence import ROOT, section

ALGORITHMS = [('MORPHICA', 'MORPHICA'), ('ML-DAGL', 'ML-DAGL'), ('DEMD', 'DEMD'),
              ('CEN', 'CEN'), ('CR-MGC', 'CR-MGC'), ('GDR-TS', 'GDR_TS'),
              ('Hybrid-MADDPG', 'Hybrid_MADDPG_APF'), ('HERO', 'HERO'), ('SIDR', 'SIDR')]
HORIZONS = {100: 600, 200: 800, 500: 1280, 1000: 1800, 2000: 2560}
OUT = ROOT / 'Paper' / 'tables'
AUDIT = {'estimand': 'RMST = mean(min(T, cutoff)); administrative censoring only',
         'historical_cutoff': 'T_max - 1; sentinel cannot distinguish a last-index event',
         'bootstrap_resamples': 2000, 'bootstrap_seed': 20261009,
         'sources': {}, 'tables': {}}
RNG = np.random.default_rng(20261009)


def source(path):
    AUDIT['sources'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def summary(times, cutoff, success=None):
    values = np.asarray(times, dtype=float)
    event = values < cutoff if success is None else np.asarray(success, dtype=bool)
    observed = np.where(event, np.minimum(values, cutoff), cutoff)
    samples = observed[RNG.integers(0, len(observed), (2000, len(observed)))].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    failed = int((~event).sum()); n = len(values); z = 1.959963984540054
    p = failed / n; den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    delta = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return {'n': n, 'cutoff': cutoff, 'rmst': float(observed.mean()),
            'rmst_ci': [float(low), float(high)], 'median': float(np.median(observed)),
            'q25': float(np.percentile(observed, 25)), 'q75': float(np.percentile(observed, 75)),
            'successful_mean': float(values[event].mean()) if event.any() else None,
            'failures': failed, 'failure_ci': [max(0., centre-delta), min(1., centre+delta)]}


def history(n, d, label, dirname):
    path = ROOT / f'Logs/algorithm_sweep/N{n}/{dirname}/{dirname}_d{d}_50cases.txt'
    if not path.exists():
        return None
    source(path)
    cases = section(path, 'case'); values = section(path, 'connect_step')
    if len(cases) != 50 or len(set(cases)) != 50 or len(values) != 50:
        raise ValueError(f'Incomplete or duplicate historical cell: {path}')
    return {'N': n, 'D': d, 'method': label, 'cases': cases,
            **summary(values, HORIZONS[n]-1)}


def display(r, prefix):
    ci = r['rmst_ci']; wi = r['failure_ci']
    return prefix + [f"{r['rmst']:.1f} [{ci[0]:.1f},{ci[1]:.1f}]",
                     f"{r['median']:.1f} [{r['q25']:.1f},{r['q75']:.1f}]",
                     f"{r['failures']}/{r['n']}",
                     f"{100*wi[0]:.1f}--{100*wi[1]:.1f}"]


def table(key, label, caption, headings, rows, stats):
    AUDIT['tables'][key] = stats
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'{key}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(headings); writer.writerows(rows)
    escaped = [[str(x).replace('%', r'\%').replace('_', r'\_') for x in r] for r in rows]
    alignment = 'l' * len(headings)
    header = ' & '.join(headings) + r' \\'
    text = '\n{\\scriptsize\\setlength{\\tabcolsep}{1.2pt}\n'
    text += '\\renewcommand{\\tablename}{Supplementary Table}\n\\setlength{\\LTcapwidth}{\\textwidth}\n'
    text += rf'\begin{{longtable}}{{{alignment}}}' + '\n'
    text += rf'\caption{{{caption}}}\label{{{label}}}\\' + '\n\\toprule\n' + header
    text += '\n\\midrule\n\\endfirsthead\n\\toprule\n' + header
    text += '\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n'
    text += '\n'.join(' & '.join(r) + r' \\' for r in escaped)
    return text + '\n\\end{longtable}\n}\n'


def insert(text, key, block):
    start = f'% BEGIN GENERATED {key}\n'; end = f'% END GENERATED {key}'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f'Missing or repeated insertion marker {key}')
    left, tail = text.split(start, 1); _, right = tail.split(end, 1)
    return left + start + block + end + right


def experiment_rows(suite):
    path = ROOT / f'Logs/manuscript_revision/{suite}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    source(path)
    rows = list(csv.DictReader(path.open()))
    tasks = {(r['N'], r['D'], r['case'], r['variant']) for r in rows}
    expected = {'ablation': 100, 'nonideal': 750, 'null': 300, 'relay': 450}[suite]
    if len(rows) != expected or len(tasks) != expected:
        raise ValueError(f'Incomplete {suite}: {len(rows)}/{expected}')
    return rows


def build_main():
    cells = []
    for d in range(50, 1000, 50):
        reference = None
        for label, dirname in ALGORITHMS:
            r = history(1000, d, label, dirname)
            if r is None:
                raise FileNotFoundError(f'Flagship cell absent: {label}, D={d}')
            if reference is None:
                reference = r['cases']
            if r['cases'] != reference:
                raise ValueError(f'Unmatched flagship cell: {label}, D={d}')
            cells.append(r)
    blocks = table('main_n1000', 'tab:supp_main_n1000',
        r'Complete $N=1000$ benchmark: 19 damage levels and nine methods, 50 matched cases per cell. '
        r'RMST is restricted to the recording cutoff $1799$; horizon sentinels are right-censored. '
        r'Brackets give bootstrap 95\% intervals for RMST and quartiles for the median; failure intervals are Wilson 95\% percentages.',
        ['$D$', 'Method', 'RMST [95\% CI]', 'Median [Q1,Q3]', 'Failures', 'Failure CI (\%)'],
        [display(r, [r['D'], r['method']]) for r in cells], cells)
    aggregate = []
    for label, dirname in ALGORITHMS:
        times = []
        for d in range(50,1000,50):
            times.extend(section(ROOT/f'Logs/algorithm_sweep/N1000/{dirname}/{dirname}_d{d}_50cases.txt', 'connect_step'))
        aggregate.append({'method': label, **summary(times,1799)})
    blocks += table('main_aggregate','tab:supp_main_aggregate',
        r'Aggregate $N=1000$ restricted mean recovery and censoring counts over 950 cases per method; cutoff 1799 steps.',
        ['Method','RMST [95\% CI]','Median [Q1,Q3]','Failures','Failure CI (\%)'],
        [display(r,[r['method']]) for r in aggregate],aggregate)
    return blocks


def build_ablation():
    global RNG
    previous_rng = RNG
    # The repaired cohort's intervals are independent of which other tables
    # are rebuilt in the same invocation.
    RNG = np.random.default_rng(20261009)
    rows = experiment_rows('ablation'); cells=[]
    previous = json.loads((OUT/'supplement_audit.json').read_text())['tables']['ablations']
    from repair_ablation_provenance import validate
    repair_path = ROOT/'Logs/manuscript_revision/ablation_repair.csv'
    manifest_path = repair_path.with_suffix('.manifest.json')
    repair = list(csv.DictReader(repair_path.open()))
    validate(repair)
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(repair_path.read_bytes()).hexdigest() != manifest['csv_sha256']:
        raise ValueError('Reconstructed ablation CSV changed')
    for name, digest in manifest['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Reconstructed ablation source changed: {name}')
    source(repair_path); source(manifest_path)
    # Read the mapping as syntax, avoiding plotting imports and Configurations.
    import ast
    tree=ast.parse((ROOT/'Paper/plot_fig3.py').read_text())
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='ABLATION_PATHS' for t in n.targets))
    for n in [100,200,500,1000]:
        reference=None
        reference_times=None
        for variant in ['full','no-R','no-P','no-S']:
            if n==1000:
                if variant in ['no-P','no-S']:
                    rs=sorted((r for r in rows if r['variant']==variant), key=lambda r:int(r['case']))
                    cases=[int(r['case']) for r in rs]
                    values=[int(r['T_conn']) for r in rs]
                    events=[int(r['success']) for r in rs]
                else:
                    folder = 'MORPHICA' if variant == 'full' else 'MORPHICA_no-R'
                    path=ROOT/f'Logs/algorithm_sweep/N1000/{folder}/MORPHICA_d500_50cases.txt'
                    source(path)
                    cases=section(path,'case'); values=section(path,'connect_step')
                    events=[int(t<1799) for t in values]
                if cases != list(range(50)):
                    raise ValueError('Offset-zero cohort changed')
                rs=sorted((r for r in repair if r['variant']==variant), key=lambda r:int(r['case']))
                cases += [int(r['case']) for r in rs]
                values += [int(r['T_conn']) for r in rs]
                events += [int(r['success']) for r in rs]
                s=summary(values,1799,events)
                export=ROOT/mapping[n][variant]
                export.parent.mkdir(parents=True,exist_ok=True)
                export.write_text('case:\n '+repr(cases)+'\n\nconnect_step:\n '+
                    repr(values)+ '\n\nsource: combined retained and reconstructed ablation cohorts\n')
                source(export)
            else:
                path=ROOT/mapping[n][variant];source(path)
                cases=section(path,'case');values=section(path,'connect_step')
                s=summary(values,HORIZONS[n]-1)
            capped=np.minimum(values,HORIZONS[n]-1)
            if reference is None:
                reference=cases
                reference_times=capped
            if cases!=reference:raise ValueError(f'Unmatched ablation N={n}, {variant}')
            delta=capped-reference_times
            sampled=delta[RNG.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
            s['paired_delta']=float(delta.mean())
            s['paired_delta_ci']=[float(v) for v in np.percentile(sampled,[2.5,97.5])]
            if n < 1000:
                prior = next(r for r in previous if r['N']==n and r['variant']==variant)
                if prior['cases'] != cases or prior['rmst'] != s['rmst'] or prior['paired_delta'] != s['paired_delta']:
                    raise ValueError('Unchanged scale cell differs from recorded statistics')
                s['rmst_ci'] = prior['rmst_ci']
                s['paired_delta_ci'] = prior['paired_delta_ci']
            cells.append({'N':n,'variant':variant,'cases':cases,**s})
    RNG = previous_rng
    AUDIT['ablation_protocol'] = {
        'N1000_per_variant': 100,
        'retained_configurations': 50,
        'reconstructed_configurations': 50,
        'reconstructed_rows': len(repair),
        'legacy_outcomes_reproduced': {
            variant: item['matching_outcomes']
            for variant, item in manifest['legacy_reconstruction'].items()},
        'diagnostic': 'Logs/manuscript_revision/diagnostics/case129_replay.json',
        'cohort_summary': 'Logs/manuscript_revision/ablation_cohorts.json',
        'case_ids_and_seeds_in_manuscript': False,
    }
    return table('ablations','tab:supp_ablations',
        r'Operator ablations at 50\% node loss: 50 configurations per variant at $N=100,200,500$ '
        r'and 100 at $N=1000$. All variants use the same initial configurations and damage. '
        r'Cutoffs are 599, 799, 1279 and 1799 for $N=100,200,500,1000$. '
        r'Changes are variant minus full RMST. RMST intervals use marginal bootstrap resampling; '
        r'$\Delta$RMST intervals use paired resampling of the matched configurations.',
        ['$N$','Variant','RMST [95\% CI]','$\Delta$RMST [95\% CI]','Failures','Failure CI (\%)'],
        [[r['N'],r['variant'],display(r,[])[0],
          f"{r['paired_delta']:.1f} [{r['paired_delta_ci'][0]:.1f},{r['paired_delta_ci'][1]:.1f}]",
          display(r,[])[2],display(r,[])[3]] for r in cells],cells)


def build_ugv():
    path=ROOT/'Logs/ugv/manuscript_matrix/metrics/ugv_summary.csv';source(path)
    rows=list(csv.DictReader(path.open()));cells=[]
    frozen_rows=rows
    for terrain in ['ridge','rolling','roughland','mixed']:
        for d in [30,50]:
            reference=None
            for method in ['MORPHICA','CEN','DEMD','CR-MGC','ML-DAGL','HERO']:
                rs=sorted((r for r in rows if r['terrain']==terrain and int(r['D'])==d and r['algorithm']==method),key=lambda r:int(r['seed_index']))
                cases=[(r['scenario_id'],r['destroy_list_hash']) for r in rs]
                if len(rs)!=25:raise ValueError(f'UGV cell incomplete {terrain}/{d}/{method}')
                if reference is None:reference=cases
                if cases!=reference:raise ValueError('UGV scenario or mask mismatch')
                s=summary([int(r['T_conn']) if r['success']=='True' else 600 for r in rs],600,[r['success']=='True' for r in rs])
                cells.append({'terrain':terrain,'D':d,'method':method,**s})
    block=table('ugv_cells','tab:supp_ugv_cells',
        r'UGV terrain-by-damage cells: 25 paired scenarios per cell, six methods, cutoff 600 steps. '
        r'Failures and Wilson intervals accompany RMST.',
        ['Terrain','$D$','Method','RMST [95\% CI]','Median [Q1,Q3]','Failures','Failure CI (\%)'],
        [display(r,[r['terrain'],r['D'],r['method']]) for r in cells],cells)
    path=ROOT/'Logs/ugv/cr80_sensitivity/metrics/ugv_summary.csv';source(path)
    cr80_rows=list(csv.DictReader(path.open()));cells=[]
    selected={r['scenario_id'] for r in frozen_rows if r['terrain']=='ridge'}
    AUDIT['ugv_cr_selection']={
        'rule':'retain exactly the frozen ridge scenario IDs at both communication radii',
        'scenario_ids':sorted(selected), 'cr80_raw_rows':len(cr80_rows)}
    for cr in [80,120]:
        for d in [30,50]:
            for method in ['MORPHICA','CEN','DEMD']:
                available=cr80_rows if cr==80 else frozen_rows
                rs=[r for r in available if r['scenario_id'] in selected and float(r['communication_range'])==cr
                    and int(r['D'])==d and r['algorithm']==method]
                if len(rs)!=25:raise ValueError(f'UGV CR sensitivity incomplete {cr}/{d}/{method}')
                reference={r['scenario_id']:r for r in frozen_rows if r['terrain']=='ridge'
                           and int(r['D'])==d and r['algorithm']==method}
                for row in rs:
                    for field in ['destroy_list_hash','seed_value','terrain_seed','dt','v_max','omega_max']:
                        if row[field]!=reference[row['scenario_id']][field]:
                            raise ValueError(f'UGV CR paired input mismatch: {field}')
                cells.append({'CR':cr,'D':d,'method':method,**summary(
                    [int(r['T_conn']) if r['success']=='True' else 600 for r in rs],
                    600,[r['success']=='True' for r in rs])})
    block+=table('ugv_cr','tab:supp_ugv_cr',r'Matched ridge-only UGV communication-range sensitivity, 25 scenarios per damage level and method; cutoff 600 steps.',
        ['CR','$D$','Method','RMST [95\% CI]','Median [Q1,Q3]','Failures','Failure CI (\%)'],
        [display(r,[r['CR'],r['D'],r['method']]) for r in cells],cells)
    return block


def build_experiment(suite):
    rows=experiment_rows(suite);cells=[]
    for d in sorted({int(r['D']) for r in rows}):
        variants=sorted({r['variant'] for r in rows if int(r['D'])==d})
        reference=None
        for variant in variants:
            rs=sorted((r for r in rows if int(r['D'])==d and r['variant']==variant),key=lambda r:int(r['case']))
            cases=[(r['case'],r['mask_sha256']) for r in rs]
            if reference is None:reference=cases
            if cases!=reference:raise ValueError(f'Unmatched {suite} cell')
            cells.append({'D':d,'variant':variant,**summary([int(r['T_conn']) for r in rs],int(rs[0]['horizon'])-1,[int(r['success']) for r in rs])})
    if suite=='null':
        for d in sorted({int(r['D']) for r in rows}):
            cell=history(1000,d,'MORPHICA','MORPHICA')
            null_cases=sorted(int(r['case']) for r in rows if int(r['D'])==d)
            if null_cases!=cell['cases']:
                raise ValueError('Random-waypoint comparison is not matched to MORPHICA')
            cell['variant']='MORPHICA'
            cells.append(cell)
    caption={'nonideal':r'Graph-observation stress at $N=1000$: 25 identical damage configurations per condition, cutoff 1799. Noise is applied after delay; contact loss is undirected.',
             'null':r'Random-waypoint null and MORPHICA on the same 300 damage cases, 50 per damage level, $N=1000$, cutoff 1799. Each null agent independently samples a waypoint uniformly in the arena and moves at unit bounded speed without messages or a shared target.',
             'relay':r'Finite synchronous identifier-deduplicated flooding at $N=200$: 25 matched cases per damage level and relay budget, cutoff 799. The graph is held fixed during each relay block; infinite denotes exact component aggregates.'}[suite]
    return table(suite,f'tab:supp_{suite}',caption,
        ['$D$','Condition','RMST [95\% CI]','Median [Q1,Q3]','Failures','Failure CI (\%)'],
        [display(r,[r['D'],r['variant']]) for r in cells],cells)


def build_n2000():
    methods = [('MORPHICA','MORPHICA'),('ML-DAGL','ML-DAGL'),
               ('CEN','CEN'),('DEMD','DEMD'),('CR-MGC','CR-MGC')]
    reference = {}
    aggregate = []
    for label, dirname in methods:
        times = []
        cases_by_damage = {}
        for d in [500,1000,1500]:
            cell = history(2000,d,label,dirname)
            if cell is None:
                raise FileNotFoundError(f'N=2000 stress cell absent: {label}, D={d}')
            if label == 'MORPHICA':
                reference[d] = cell['cases']
            if cell['cases'] != reference[d]:
                raise ValueError(f'Unmatched N=2000 stress cell: {label}, D={d}')
            cases_by_damage[str(d)] = cell['cases']
            path = ROOT/f'Logs/algorithm_sweep/N2000/{dirname}/{dirname}_d{d}_50cases.txt'
            times.extend(section(path,'connect_step'))
        aggregate.append({'method':label,'cases_by_damage':cases_by_damage,
                          **summary(times,2559)})
    full = aggregate[0]['rmst']
    for row in aggregate:
        row['rmst_ratio_to_morphica'] = row['rmst']/full
    return table('n2000','tab:supp_n2000',
        r'Matched $N=2000$ stress benchmark: 50 cases at each of $D=500,1000,1500$, '
        r'150 per method. RMST is restricted to cutoff 2559. Ratios compare each restricted mean '
        r'with that of MORPHICA; brackets give bootstrap 95\% intervals.',
        ['Method','RMST [95\% CI]','Failures','RMST ratio'],
        [[r['method'],display(r,[])[0],f"{r['failures']}/{r['n']}",
          f"{r['rmst_ratio_to_morphica']:.2f}"] for r in aggregate],aggregate)


def order_tables_by_citation(text):
    """Keep narrative Notes stable; order the numbered table appendix by first citation."""
    main, supplementary = text.split(r'\section*{Supplementary information}',1)
    first = {}
    for match in re.finditer(r'\\ref\{(tab:supp_[^}]+)\}',main):
        first.setdefault(match.group(1),len(first))
    blocks = []
    pattern = r'% BEGIN GENERATED (\w+)\n.*?% END GENERATED \1'
    def take_generated(match):
        block = match.group(0)
        labels = re.findall(r'\\label\{(tab:supp_[^}]+)\}',block)
        blocks.append((labels,block))
        return ''
    supplementary = re.sub(pattern,take_generated,supplementary,flags=re.S)
    pattern = r'\\suppcaption\{(tab:supp_[^}]+)\}.*?\\end\{center\}'
    def take_manual(match):
        blocks.append(([match.group(1)],match.group(0)))
        return ''
    supplementary = re.sub(pattern,take_manual,supplementary,flags=re.S)
    labels = [label for group,_ in blocks for label in group]
    if len(labels) != 19 or len(set(labels)) != 19 or set(labels) != set(first):
        raise ValueError(f'Every one of 19 tables must be cited in main text: {set(labels)^set(first)}')
    blocks.sort(key=lambda item:first[item[0][0]])
    ordered = [label for group,_ in blocks for label in group]
    if ordered != list(first):
        raise ValueError('A generated table group conflicts with first-citation order')
    heading = r'\subsection*{Supplementary Tables}'
    if heading in supplementary:
        supplementary = supplementary.replace(heading,'',1)
    supplementary = re.sub(r'\n{3,}','\n\n',supplementary)
    tables = heading + '\n\n' + '\n\n'.join(block for _,block in blocks) + '\n\n'
    supplementary = supplementary.replace(r'\section*{Code availability}',tables+r'\section*{Code availability}',1)
    AUDIT['table_citation_order'] = [
        {'number':f'S{i}','label':label} for i,label in enumerate(ordered,1)]
    return main+r'\section*{Supplementary information}'+supplementary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sections', nargs='+', default=['main','ablation','ugv','nonideal','null','relay','n2000'])
    args=parser.parse_args()
    manuscript=ROOT/'Paper/sn-article.tex'
    # This code-and-data release ships without the manuscript source. Build the
    # tables and the audit file either way, and insert inline only when the
    # manuscript is present.
    text=manuscript.read_text() if manuscript.exists() else None
    for key in args.sections:
        block={'main':build_main,'ablation':build_ablation,'ugv':build_ugv,'n2000':build_n2000}.get(key,lambda:build_experiment(key))()
        if text is not None:
            text=insert(text,key,block)
    if text is not None:
        text = order_tables_by_citation(text)
        manuscript.write_text(text)
    else:
        print('Paper/sn-article.tex not present: wrote tables and audit only')
    path=OUT/'supplement_audit.json'
    if path.exists():
        old=json.loads(path.read_text());old['sources'].update(AUDIT['sources']);old['tables'].update(AUDIT['tables'])
        AUDIT['sources']=old['sources'];AUDIT['tables']=old['tables']
        if 'ablation_protocol' not in AUDIT and 'ablation_protocol' in old:
            AUDIT['ablation_protocol'] = old['ablation_protocol']
    path.write_text(json.dumps(AUDIT,indent=2)+'\n')
    print('Updated inline SI:', ', '.join(args.sections))


if __name__=='__main__':main()
