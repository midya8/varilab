"""Exact binary-rational ELBO Hessian inertia and congruence certificates."""
from fractions import Fraction as F
from .core import compile_model, means_vector, canonical, elbo_gradient


def _pack(x):
    return {'numerator':x.numerator,'denominator':x.denominator}


def _matrix(a):
    return [[_pack(v) for v in row] for row in a]


def _inertia(matrix):
    """Symmetric congruence with 1x1 pivots, zero-diagonal 2x2 pivots."""
    n=len(matrix);a=[row[:] for row in matrix]
    t=[[F(i==j) for j in range(n)] for i in range(n)]
    counts={'positive':0,'negative':0,'zero':0};blocks=[];witness={}
    def swap(i,j):
        a[i],a[j]=a[j],a[i]
        for row in a:row[i],row[j]=row[j],row[i]
        for row in t:row[i],row[j]=row[j],row[i]
    def record(sign,v):
        if sign not in witness:
            witness[sign]=[sum(t[i][j]*v.get(j,F(0)) for j in v) for i in range(n)]
    k=0
    while k<n:
        p=next((i for i in range(k,n) if a[i][i]),None)
        if p is not None:
            swap(k,p);b=a[k][k];size=1;inv=[[1/b]]
            sign='positive' if b>0 else 'negative';counts[sign]+=1;record(sign,{k:F(1)})
        else:
            pair=next(((i,j) for i in range(k,n) for j in range(i+1,n) if a[i][j]),None)
            if pair is None:
                counts['zero']+=n-k
                for i in range(k,n):blocks.append({'offset':i,'size':1})
                break
            i,j=pair;swap(k,i);swap(k+1,j);b=a[k][k+1];size=2
            inv=[[F(0),1/b],[1/b,F(0)]]
            counts['positive']+=1;counts['negative']+=1
            sign=F(1 if b>0 else -1)
            record('positive',{k:F(1),k+1:sign});record('negative',{k:F(1),k+1:-sign})
        blocks.append({'offset':k,'size':size})
        c=[[a[k+i][j] for j in range(k+size,n)] for i in range(size)]
        v=[[sum(inv[i][r]*c[r][j] for r in range(size)) for j in range(n-k-size)] for i in range(size)]
        old=[row[k:k+size] for row in t]
        for row in range(n):
            for j in range(k+size,n):t[row][j]-=sum(old[row][r]*v[r][j-k-size] for r in range(size))
        for i in range(k+size,n):
            for j in range(k+size,n):a[i][j]-=sum(c[r][i-k-size]*v[r][j-k-size] for r in range(size))
        for i in range(k,k+size):
            for j in range(k+size,n):a[i][j]=a[j][i]=F(0)
        k+=size
    return a,t,counts,blocks,witness


def curvature(payload,means):
    """Audit the mathematical Hessian at exactly represented interior means.

    No float eigensolver or tolerance decides a curvature sign. The supplied
    numeric fields are compiled to binary floats, then interpreted as rationals.
    This does not certify stationarity, a maximum, or model accuracy.
    """
    h,e=compile_model(payload);m=means_vector(means,len(h));n=len(h)
    if any(abs(v)==1 for v in m):raise ValueError('curvature requires interior means')
    r=list(map(F,m));a=[[F(0) for _ in h] for _ in h]
    for i in range(n):a[i][i]=-1/(1-r[i]*r[i])
    for i,j,w in e:a[i][j]=a[j][i]=F(w)
    d,t,inertia,blocks,witness=_inertia(a)
    p,q,z=(inertia[v] for v in ('positive','negative','zero'))
    classification=('indefinite' if p and q else 'negative_definite' if q==n else
                    'positive_definite' if p==n else 'negative_semidefinite' if q else
                    'positive_semidefinite' if p else 'zero')
    directions={}
    for sign,v in witness.items():
        value=sum(v[i]*a[i][j]*v[j] for i in range(n) for j in range(n))
        directions[sign]={'direction':[_pack(x) for x in v],'quadratic_form':_pack(value)}
    return {'format':'varilab-curvature-v1','model':canonical(h,e),'means':list(m),
            'hessian':_matrix(a),'congruence':_matrix(t),'block_diagonal':_matrix(d),
            'pivot_blocks':blocks,'inertia':inertia,'classification':classification,
            'curvature_witnesses':directions,'gradient_float':elbo_gradient(payload,m),
            'certificate_identity':'transpose(congruence) * hessian * congruence = block_diagonal',
            'scope':'exact rational Hessian at represented binary-float means and couplings; congruence is invertible by construction; curvature alone does not certify stationarity, an optimum, marginal accuracy or uncertainty calibration'}


def curvature_demo():
    """Direct boundary neighbors plus the existing coupled fixture, no search."""
    import math
    rows=[]
    for name,j in [('below critical',math.nextafter(1,0)),('critical',1.0),('above critical',math.nextafter(1,2)),('symmetric J=3',3.0),('broken symmetry J=3',3.0)]:
        model={'version':1,'fields':[0,0],'edges':[{'i':0,'j':1,'weight':j}]}
        means=[.9949015284526289]*2 if name.startswith('broken') else [0,0]
        audit=curvature(model,means)
        # Transcendental libm gradient diagnostics are not rational certificates.
        audit.pop('gradient_float')
        rows.append({'name':name,'audit':audit})
    assert [r['audit']['classification'] for r in rows]==['negative_definite','negative_semidefinite','indefinite','indefinite','negative_definite']
    return {'rows':rows,'selection':'direct critical J=1 binary-float neighbors, plus the existing J=3 symmetric point and captured binary-float broken-symmetry means 0.9949015284526289; no parameter search',
            'scope':'exact Hessian inertia at fixed represented means; libm-dependent float gradient omitted from this reproducible certificate fixture, available from curvature API; not exact stationarity or an optimum certificate; inertia counts signs, not eigenvalue magnitudes'}
