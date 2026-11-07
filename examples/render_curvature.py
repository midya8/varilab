"""Exact inertia counts, not eigenvalue estimates; embeds complete certificates."""
import argparse,html,json
from pathlib import Path
from varilab.curvature import curvature_demo


def render(payload):
    if payload!=curvature_demo():raise ValueError('requires complete unchanged curvature demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650">',
         '<rect width="1120" height="650" fill="#101827"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>',
         '<metadata>'+html.escape(json.dumps(payload,sort_keys=True,allow_nan=False))+'</metadata>']
    def text(x,y,s,size=15,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(35,43,'VariLab | coordinate stopping is not a maximum certificate',27)
    text(35,75,'Exact-rational Hessian inertia separates a saddle, a singular boundary and strict negative curvature.',17)
    text(35,103,'H_ii = -1 / (1 - m_i^2), H_ij = J_ij. Binary floats become exact rationals before Hessian arithmetic.',15,'#b7c4d9')
    for x,label in [(35,'Supplied point / coupling'),(400,'Means'),(595,'Positive'),(745,'Negative'),(895,'Zero')]:text(x,150,label,15)
    colors=['#efbd68','#63d8bc','#a5b8f9'];keys=['positive','negative','zero']
    for row,item in enumerate(payload['rows']):
        a=item['audit'];y=190+75*row
        text(35,y,item['name'],16);text(35,y+23,'J = '+repr(a['model']['edges'][0]['weight']),14,'#b7c4d9')
        text(400,y,'('+', '.join(f'{m:.6f}' for m in a['means'])+')',14)
        text(400,y+23,a['classification'].replace('_',' '),13,'#b7c4d9')
        for col,key in enumerate(keys):
            value=a['inertia'][key];x=595+150*col
            out.append(f'<rect data-row="{row}" data-sign="{key}" data-value="{value}" x="{x}" y="{y-13}" width="{45*value}" height="16" fill="{colors[col]}"/>')
            text(x,y+26,str(value),15,colors[col])
    text(35,574,'Counts share one scale: 45 pixels per eigenvalue. Congruence preserves signs, not eigenvalue magnitudes.',15,'#b7c4d9')
    text(35,602,'No eigenvalue threshold: the two neighbors of J=1 have different certified signs. Full rational certificates embedded.',15,'#b7c4d9')
    text(35,630,'Curvature alone does not certify stationarity, an optimum, marginal accuracy or calibrated uncertainty.',15,'#b7c4d9')
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'assets'/'curvature-audit.svg';s=render(curvature_demo())
    if args.check:
        if path.read_text()!=s:raise SystemExit('curvature SVG differs')
    else:path.write_text(s)
