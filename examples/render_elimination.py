"""Exact symbolic bucket-size geometry; portable rounded numeric reference labels."""
import argparse,html,json,math
from pathlib import Path
from varilab.elimination import elimination_demo


def render(payload):
    if payload!=elimination_demo():raise ValueError('requires complete unchanged elimination demo')
    plans=[payload['hub_first_plan'],payload['min_fill_star']['plan'],payload['cycle_query']['plan']]
    fixed={'plans':plans,'models':[payload['min_fill_star']['model'],payload['cycle_query']['model']],
           'gates':payload['gates'],'selection':payload['selection'],'scope':payload['scope'],
           'analytic_star_log_partition':round(payload['analytic_star_log_partition'],12),
           'analytic_cycle_log_partition':round(payload['analytic_cycle_log_partition'],12),
           'analytic_cycle_edge_moment':round(payload['analytic_cycle_edge_moment'],12),
           'cycle_query_probabilities':[round(r['probability'],12) for r in payload['cycle_query']['query_states']],
           'display_scope':'full exact symbolic plans and counters; analytic/numeric labels rounded to 12 decimals for portable display; full floating factor tables remain in elimination-demo, numerically checked rather than byte-frozen'}
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="710" viewBox="0 0 1120 710">',
         '<rect width="1120" height="710" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>',
         '<metadata>'+html.escape(json.dumps(fixed,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=15,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(35,42,'VariLab | a bad elimination order is not an impossible graph',27)
    text(35,76,'Same 16-spin star partition query: hub first is refused; min-fill fits. A 64-spin cycle also fits.',17)
    text(35,112,'log2(peak bucket assignments), not runtime. Each vertical unit doubles the assignment count.',16)
    for exponent in range(0,17,4):
        y=450-18*exponent;out.append(f'<path d="M 80 {y} H 1070" stroke="#33445e"/>');text(38,y+5,str(exponent),13)
    out.append('<path data-cap="4096" d="M 80 234 H 1070" stroke="#efbd68" stroke-dasharray="6 5"/>');text(690,224,'cap: 4096 assignments (2^12)',15,'#efbd68')
    for i,(plan,label) in enumerate(zip(plans,['16-star, hub first','16-star, min-fill','64-cycle, min-fill; query [0,1]'])):
        count=plan['peak_bucket_states'];exponent=count.bit_length()-1;x=170+330*i;y=450-18*exponent;color='#63d8bc' if plan['accepted'] else '#e58f99'
        out.append(f'<rect data-plan="{i}" data-states="{count}" x="{x}" y="{y}" width="84" height="{18*exponent}" fill="{color}"/>')
        text(x-10,y-13,str(count)+' states',17,color);text(x-70,482,label,16);text(x-15,509,'ACCEPTED' if plan['accepted'] else 'REJECTED',15,color)
    text(35,550,'All factor scopes and work counters are planned before numeric table allocation. A refusal returns no partial inference.',16)
    text(35,582,'Analytic log Z: star '+f"{fixed['analytic_star_log_partition']:.6f}"+', cycle '+f"{fixed['analytic_cycle_log_partition']:.6f}"+'. The accepted cyclic query matches its separate ring formula.',16)
    text(35,614,'Cycle edge E[z0*z1] = '+f"{fixed['analytic_cycle_edge_moment']:.6f}"+'. All four conditional query probabilities and intermediate factors are returned.',16)
    text(35,647,'Min-fill is a deterministic heuristic, not minimum treewidth. Caps count assignments/reads/entries, not seconds or memory bytes.',15,'#b7c4d9')
    text(35,675,'64-spin graph cap, 8 query spins; float log-sum-exp, not rational arithmetic, causal inference or a speedup benchmark.',15,'#b7c4d9')
    text(35,700,'Exact symbolic metadata; float labels rounded to 12 decimals. Full numerical factor traces remain in the demo.',14,'#b7c4d9')
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();path=Path(__file__).resolve().parents[1]/'assets'/'elimination-audit.svg';s=render(elimination_demo())
    if a.check:
        if path.read_text()!=s:raise SystemExit('elimination SVG differs')
    else:path.write_text(s)
