"""Fixed analytic four-node conditioning figure with explicit rounded display metadata."""
import argparse,html,json
from pathlib import Path
from varilab.forest import forest_demo


def render(payload):
    if payload!=forest_demo():raise ValueError('requires complete unchanged forest demo')
    # Float transcendental messages are checked numerically, not byte-frozen.
    # Embed only the fixed small-model inputs and analytic displayed values.
    fixed={'model':payload['prior']['model'],'evidence':payload['conditioned']['evidence'],
           'analytic_means':[round(x,12) for x in payload['analytic_conditional_means']],
           'gates':payload['gates'],'selection':payload['selection'],
           'scope':'figure uses analytic rounded display values; full float messages remain in forest-demo; exact structure does not imply rational arithmetic'}
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="660" viewBox="0 0 1120 660">',
         '<rect width="1120" height="660" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>',
         '<metadata>'+html.escape(json.dumps(fixed,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=15,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(35,43,'VariLab | evidence travels through an exact tree',28)
    text(35,75,'Four-node zero-field Ising chain, J=2. Prior means are all zero. Observe spin 0 = +1.',17)
    for i in range(3):out.append(f'<path d="M {160+265*i} 143 H {345+265*i}" stroke="#879cbd" stroke-width="3"/>')
    for i in range(4):
        x=120+265*i;out.append(f'<circle cx="{x}" cy="143" r="28" fill="'+('#63d8bc' if i==0 else '#33445e')+'"/>');text(x-6,149,str(i),18,'#101827' if i==0 else '#e7edf8')
    text(35,213,'Mean spin, one shared vertical scale',16)
    for v in (0,.25,.5,.75,1):
        y=465-190*v;out.append(f'<path d="M 80 {y} H 1070" stroke="#33445e"/>');text(35,y+5,str(v),13)
    values=fixed['analytic_means']
    for i,v in enumerate(values):
        x=105+265*i;y=465-190*v
        out.append(f'<rect data-node="{i}" data-value="{v}" x="{x}" y="{y}" width="64" height="{190*v}" fill="#63d8bc"/>')
        text(x-8,y-12,f'{v:.6f}',16,'#63d8bc');text(x-4,493,'spin '+str(i),15)
        out.append(f'<circle data-prior-node="{i}" data-prior-mean="0" cx="{x+32}" cy="465" r="5" fill="#efbd68"/>')
    text(35,535,'Green: conditioned mean tanh(2)^distance. Amber baseline: prior mean 0, a different query.',16)
    text(35,565,'P(spin 0 = +1) = 0.5. Endpoint > 0.8 gate passes with evidence and fails without it.',16)
    text(35,595,'128-node chain: 254 directed messages, no full-state enumeration. Only acyclic graphs are accepted.',15,'#b7c4d9')
    text(35,623,'Supplied-model conditioning, not causal intervention. Float log-sum-exp; possible underflow retains finite logs.',15,'#b7c4d9')
    text(35,649,'Display geometry uses analytic values rounded to 12 decimals; full float messages are returned by the demo.',14,'#b7c4d9')
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();path=Path(__file__).resolve().parents[1]/'assets'/'forest-audit.svg';s=render(forest_demo())
    if args.check:
        if path.read_text()!=s:raise SystemExit('forest SVG differs')
    else:path.write_text(s)
