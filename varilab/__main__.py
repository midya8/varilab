import argparse
import json
from .core import exact,mean_field,compare
from .demo import demo
from .curvature import curvature,curvature_demo


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def load(path):
    with open(path,'rb') as f:raw=f.read(262145)
    if len(raw)>262144:raise ValueError('JSON files are limited to 256 KiB')
    return json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def main(argv=None):
    p=argparse.ArgumentParser(description='Exact enumeration versus product mean-field inference in tiny Ising models')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('demo');sub.add_parser('curvature-demo')
    a=sub.add_parser('exact');a.add_argument('model')
    a=sub.add_parser('curvature');a.add_argument('model');a.add_argument('--means',required=True,help='JSON array of interior spin means')
    for name in ('fit','compare'):
        a=sub.add_parser(name);a.add_argument('model');a.add_argument('--initial',help='JSON array of initial spin means')
        a.add_argument('--order',help='JSON array of coordinate indices')
        a.add_argument('--sweeps',type=int,default=100);a.add_argument('--tolerance',type=float,default=1e-10)
    args=p.parse_args(argv)
    try:
        if args.command=='demo':out=demo()
        elif args.command=='curvature-demo':out=curvature_demo()
        elif args.command=='exact':out=exact(load(args.model))
        elif args.command=='curvature':out=curvature(load(args.model),json.loads(args.means))
        else:
            kw={'initial':json.loads(args.initial) if args.initial is not None else None,
                'order':json.loads(args.order) if args.order is not None else None,
                'sweeps':args.sweeps,'tolerance':args.tolerance}
            out=(mean_field if args.command=='fit' else compare)(load(args.model),**kw)
        text=json.dumps(out,indent=2,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,'error: '+str(e)+'\n')
    print(text)


if __name__=='__main__':main()
