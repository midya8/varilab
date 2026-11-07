"""All state probabilities and objective/mean errors from the unchanged demo."""
import argparse
import html
import json
from pathlib import Path
from varilab.demo import demo


def render(payload):
    if payload!=demo():raise ValueError('requires complete unchanged demo')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="740" viewBox="0 0 1120 740">',
         '<rect width="1120" height="740" fill="#101827"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>',
         '<metadata>'+html.escape(json.dumps(payload,sort_keys=True,allow_nan=False))+'</metadata>']
    def text(x,y,s,size=15,color=None):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}"'+(f' style="fill:{color}"' if color else '')+'>'+html.escape(s)+'</text>')
    text(35,42,'VariLab | a better ELBO can give worse marginal means',28)
    text(35,72,'Two spins with zero fields and J = 3. Exact means are (0, 0); exact cross-covariance is tanh(3) = 0.995055.',16)
    colors=['#63d8bc','#9cb5ff','#efc76a','#f2a7b5'];names=['Exact target','Symmetric mean-field','Positive mean-field','Negative mean-field']
    for i,(color,name) in enumerate(zip(colors,names)):text(60+270*i,110,name,15,color)
    for v in (0,.25,.5,.75,1):
        y=390-230*v;svg.append(f'<path d="M 80 {y} H 1060" stroke="#33445e"/>');text(36,y+5,str(v),13)
    arms=payload['coupled_arms'];target=arms[0]['comparison']['exact']['states']
    for j,state in enumerate(target):
        text(160+245*j,420,str(tuple(state['state'])),15)
        values=[state['probability']]+[a['comparison']['q_states'][j]['q_probability'] for a in arms]
        for i,value in enumerate(values):
            x=112+245*j+34*i;y=390-230*value
            svg.append(f'<rect data-state="{j}" data-arm="{i}" data-value="{value}" x="{x}" y="{y}" width="27" height="{230*value}" fill="{colors[i]}"/>')
    text(35,460,'Product q cannot represent the target correlation. Symmetry-breaking chooses opposite modes from opposite starts.',16)
    for x,s in [(45,'Start / result'),(400,'ELBO (higher is better)'),(650,'KL(q || p) (lower is better)'),(925,'Max mean error')]:text(x,505,s,14)
    for i,a in enumerate(arms):
        c=a['comparison'];y=540+38*i
        text(45,y,a['name']+'; stopping rule met',16)
        for key,x,value in [('elbo',400,c['mean_field']['elbo']),('reverse_kl',650,c['reverse_kl_direct']),('mean_error',925,c['max_mean_error'])]:
            text(x,y,f'{value:.6f}',17);svg.append(f'<g data-row="{i}" data-metric="{key}" data-value="{value}"/>')
    text(35,657,'Zero means stop after one sweep at a coordinate fixed point, but a joint perturbation raises ELBO: not a maximum.',15,'#b7c4d9')
    text(35,686,'Independent-control mean-error gate passes; positive-start coupled gate fails. Full model, states and histories embedded.',15,'#b7c4d9')
    text(35,715,'Hand-specified finite-model example, not an empirical benchmark. Exact enumeration uses floating-point log-sum-exp.',15,'#b7c4d9')
    svg.append('</svg>');return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'assets'/'variational-audit.svg';actual=render(demo())
    if a.check:
        if not path.exists() or path.read_text()!=actual:raise SystemExit('figure is stale')
    else:path.write_text(actual)


if __name__=='__main__':main()
