"""Full exact-rational certificates with common-zero signed-score geometry."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from varilab.cyclic_map import cyclic_map_demo


def render(payload):
    if payload!=cyclic_map_demo():raise ValueError('requires unchanged complete cyclic MAP demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="700" viewBox="0 0 1120 700">','<rect width="1120" height="700" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(payload,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(35,43,'VariLab | local optimal choices need a compatible cyclic traceback',26)
    text(35,79,'A zero-field antiferromagnetic triangle: every singleton allows both spins, but (-1,-1,-1) is not optimal.',16)
    text(35,112,'Exact rational unnormalized scores of the same supplied model. Common zero; higher is better.',16)
    for value in [-3,-2,-1,0,1,2]:
        y=340-50*value;out.append(f'<path d="M 80 {y} H 610" stroke="#71839d" stroke-width="{2 if value==0 else 1}"/>');text(42,y+5,str(value),14)
    values=[payload['naive_score'],payload['triangle']['optimal_score']]
    for i,(d,label) in enumerate(zip(values,['Independent first spins','Compatible traceback'])):
        value=F(d['numerator'],d['denominator']);x=150+270*i;y=340-50*max(value,0);height=50*abs(value);color='#e58f99' if i==0 else '#63d8bc'
        out.append(f'<rect data-score="{value}" x="{x}" y="{y}" width="100" height="{height}" fill="{color}"/>');text(x-5,float(y)-12,str(value),20,color);text(x-60,530,label,16);text(x-24,556,'FAIL' if i==0 else 'PASS',16,color)
    text(665,183,'Separate odd-ring analytic check',21)
    text(665,225,'63 spins, all edge weights -1',17)
    text(665,263,'Optimal score = 63 - 2 = 61',19,'#63d8bc')
    text(665,301,'Optimum count = 2 x 63 = 126',19,'#63d8bc')
    text(665,339,'One unsatisfied edge; global flip.',16)
    text(665,377,'Peak bucket assignments: 8',17)
    text(665,415,'No timing or optimal-treewidth claim.',15,'#b7c4d9')
    text(665,453,'Triangle: score 1, six tied optima.',17)
    text(665,491,'Chosen witness is not a uniqueness claim.',15,'#b7c4d9')
    text(35,599,'Exact integer multiplicities, query-constrained max scores, complete factor choices and reverse context-aware traceback.',16)
    text(35,632,'Joint MAP, not marginal MAP or probabilities. Work caps do not bound variable-bit rational arithmetic or serialized bytes.',15,'#b7c4d9')
    text(35,665,'Directly constructed fixtures, not searched. Full exact demo/certificates in SVG metadata; no rounded score or tie tolerance.',15,'#b7c4d9')
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();path=Path(__file__).resolve().parents[1]/'assets'/'cyclic-map-audit.svg';s=render(cyclic_map_demo())
    if a.check:
        if path.read_text()!=s:raise SystemExit('cyclic MAP SVG differs')
    else:path.write_text(s)
