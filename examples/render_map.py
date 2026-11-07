"""MAP-versus-marginal mode counterexample with exact score certificates."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from varilab.decoding import map_demo


def render(payload):
    if payload!=map_demo():raise ValueError('requires complete unchanged MAP demo')
    fixed={k:v for k,v in payload.items() if k!='marginal_probabilities'}
    fixed['display_marginal_probabilities']=[[round(v,12) for v in row] for row in payload['marginal_probabilities']]
    fixed['display_scope']='marginal probabilities rounded to 12 decimals for portable display; every MAP score/count/backpointer remains exact binary-rational'
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="730" viewBox="0 0 1120 730">',
         '<rect width="1120" height="730" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>',
         '<metadata>'+html.escape(json.dumps(fixed,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=15,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(35,43,'VariLab | marginal modes are not a joint MAP decoder',27)
    text(35,77,'Two supplied spins: h=(1/4,63/256), J=-2. No evidence. The joint MAP is unique.',17)
    for i,row in enumerate(fixed['display_marginal_probabilities']):
        text(35+535*i,118,f'Spin {i}: P(+1)={row[1]:.6f} > P(-1)={row[0]:.6f}',17,'#b7c4d9')
    text(35,160,'Unnormalized log-potential score E(z), higher is better; one shared vertical scale',16)
    for score in (-2,-1,0,1,2):
        y=310-50*score;out.append(f'<path d="M 85 {y} H 1070" stroke="'+('#879cbd' if score==0 else '#33445e')+'"/>');text(40,y+5,str(score),13)
    optimum=F(payload['unique_case']['optimal_score']['numerator'],payload['unique_case']['optimal_score']['denominator'])
    marginal=F(payload['marginal_mode_score']['numerator'],payload['marginal_mode_score']['denominator'])
    for name,x,score,color,label in [('marginal',245,marginal,'#efbd68','Independent marginal modes (+1,+1)'),('map',745,optimum,'#63d8bc','Compatible MAP traceback (+1,-1)')]:
        value=float(score);y=min(310,310-50*value)
        out.append(f'<rect data-method="{name}" data-score="{score}" x="{x}" y="{y}" width="95" height="{50*abs(value)}" fill="{color}"/>')
        text(x-30,195 if value>0 else 437,f'{score} = {value:.6f}',16,color);text(x-155,470,label,16,color)
    text(35,507,'Exact gap: 449/128 = 3.5078125. Both gates are asserted: MAP passes, marginal-mode score fails.',16)
    text(35,548,'Ties need compatible backtracking, even when all local max-marginal choices look optimal.',17)
    text(35,580,'Symmetric J=-2, h=(0,0): 2 joint optima. Both spins individually allow -1 and +1.',16,'#b7c4d9')
    text(35,611,'Naive first local choices (-1,-1): score -2. Stored conditional traceback (-1,+1): score +2.',16)
    text(35,644,'128 isolated zero-field spins: exactly 2^128 optimal assignments, counted without enumeration.',16)
    text(35,677,'Exact rational scores/counts for compiled binary floats; ordinary marginal probabilities use float sum-product.',15,'#b7c4d9')
    text(35,708,'Constructed decision-question counterexamples, not a method ranking. Acyclic supplied graphs only, up to 128 spins.',15,'#b7c4d9')
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();path=Path(__file__).resolve().parents[1]/'assets'/'map-audit.svg';s=render(map_demo())
    if a.check:
        if path.read_text()!=s:raise SystemExit('MAP SVG differs')
    else:path.write_text(s)
