"""Small finite Ising distributions and sequential mean-field coordinate ascent."""
import itertools
import math


def number(v,name,limit=100):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit:
        raise ValueError(name+' must be finite with magnitude <= '+str(limit))
    return float(v)


def compile_model(payload):
    if not isinstance(payload,dict) or set(payload)!={'version','fields','edges'}:
        raise ValueError('model requires exactly version, fields and edges')
    if type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('version must be integer 1')
    h=payload['fields'];edges=payload['edges']
    if not isinstance(h,(list,tuple)) or not 1<=len(h)<=12:
        raise ValueError('fields needs 1..12 coordinates')
    h=tuple(number(v,'field') for v in h)
    if not isinstance(edges,(list,tuple)) or len(edges)>len(h)*(len(h)-1)//2:
        raise ValueError('edges must be a bounded list')
    found=set();out=[]
    for e in edges:
        if not isinstance(e,dict) or set(e)!={'i','j','weight'}:
            raise ValueError('each edge requires exactly i, j and weight')
        i,j=e['i'],e['j']
        if type(i) is not int or type(j) is not int or not 0<=i<j<len(h):
            raise ValueError('edges require integer 0 <= i < j < dimension')
        if (i,j) in found:raise ValueError('duplicate edge')
        found.add((i,j));out.append((i,j,number(e['weight'],'weight')))
    return h,tuple(sorted(out))


def canonical(h,edges):
    return {'version':1,'fields':list(h),'edges':[{'i':i,'j':j,'weight':w} for i,j,w in edges]}


def means_vector(values,n):
    if not isinstance(values,(list,tuple)) or len(values)!=n:
        raise ValueError('means must match dimension')
    return tuple(number(v,'mean',1) for v in values)


def _energy(h,edges,z):
    return math.fsum([a*b for a,b in zip(h,z)]+[w*z[i]*z[j] for i,j,w in edges])


def _entropy(m):
    return math.fsum(-p*math.log(p) for t in m for p in ((1+t)/2,(1-t)/2) if p>0)


def _elbo(h,edges,m):
    return _energy(h,edges,m)+_entropy(m)


def elbo(payload,means):
    h,e=compile_model(payload);m=means_vector(means,len(h))
    return {'elbo':_elbo(h,e,m),'expected_log_potential':_energy(h,e,m),'entropy':_entropy(m)}


def elbo_gradient(payload,means):
    h,e=compile_model(payload);m=means_vector(means,len(h))
    if any(abs(v)==1 for v in m):raise ValueError('ELBO gradient requires interior means')
    g=[a-math.atanh(t) for a,t in zip(h,m)]
    for i,j,w in e:g[i]+=w*m[j];g[j]+=w*m[i]
    return g


def exact(payload):
    h,e=compile_model(payload);n=len(h)
    if (1<<n)*(n+len(e)+n*n)>786432:raise ValueError('exact enumeration work exceeds 786432')
    states=list(itertools.product((-1,1),repeat=n))
    energies=[_energy(h,e,z) for z in states]
    peak=max(energies);log_z=peak+math.log(math.fsum(math.exp(v-peak) for v in energies))
    logp=[v-log_z for v in energies];p=[math.exp(v) for v in logp]
    means=[math.fsum(a*z[i] for a,z in zip(p,states)) for i in range(n)]
    second=[[math.fsum(a*z[i]*z[j] for a,z in zip(p,states)) for j in range(n)] for i in range(n)]
    return {'format':'varilab-exact-v1','model':canonical(h,e),'log_partition':log_z,
            'means':means,'second_moments':second,
            'covariance':[[v-means[i]*means[j] for j,v in enumerate(row)] for i,row in enumerate(second)],
            'states':[{'state':list(z),'log_probability':lp,'probability':a} for z,lp,a in zip(states,logp,p)],
            'state_count':len(states),'scope':'exact state enumeration with floating-point log-sum-exp; tiny supplied models only'}


def mean_field(payload,*,initial=None,sweeps=100,tolerance=1e-10,order=None):
    h,e=compile_model(payload);n=len(h)
    if type(sweeps) is not int or not 1<=sweeps<=1000:raise ValueError('sweeps must be integer 1..1000')
    tol=number(tolerance,'tolerance',1)
    if tol<0:raise ValueError('tolerance must be nonnegative')
    m=list(means_vector([0]*n if initial is None else initial,n))
    if order is None:order=list(range(n))
    if not isinstance(order,(list,tuple)) or len(order)!=n or any(type(i) is not int for i in order) or set(order)!=set(range(n)):
        raise ValueError('order must be an integer coordinate permutation')
    if sweeps*n*(n+len(e))>500000:raise ValueError('mean-field work exceeds 500000')
    adj=[[] for _ in h]
    for i,j,w in e:adj[i].append((j,w));adj[j].append((i,w))
    def residual():return max(abs(math.tanh(h[i]+math.fsum(w*m[j] for j,w in adj[i]))-m[i]) for i in range(n))
    history=[{'update':0,'sweep':0,'coordinate':None,'means':m[:],'elbo':_elbo(h,e,m),'delta':0.0}]
    converged=False;last_change=None
    for sweep in range(1,sweeps+1):
        last_change=0.0
        for i in order:
            new=math.tanh(h[i]+math.fsum(w*m[j] for j,w in adj[i]))
            change=abs(new-m[i]);m[i]=new;last_change=max(last_change,change)
            history.append({'update':len(history),'sweep':sweep,'coordinate':i,'means':m[:],'elbo':_elbo(h,e,m),'delta':change})
        if last_change<=tol:converged=True;break
    return {'format':'varilab-mean-field-v1','model':canonical(h,e),'initial':history[0]['means'][:],
            'means':m,'positive_probabilities':[(1+t)/2 for t in m],
            'elbo':_elbo(h,e,m),'expected_log_potential':_energy(h,e,m),'entropy':_entropy(m),
            'covariance':[[1-m[i]*m[i] if i==j else 0.0 for j in range(n)] for i in range(n)],
            'history':history,'sweeps_completed':sweep,'coordinate_updates':len(history)-1,
            'settings':{'max_sweeps':sweeps,'tolerance':tol,'order':list(order)},
            'stopping_rule_met':converged,'last_sweep_max_change':last_change,
            'max_coordinate_residual':residual(),
            'scope':'sequential product-Bernoulli coordinate ascent; stopping rule does not certify a local or global ELBO maximum, marginal accuracy or uncertainty calibration'}


def compare(payload,*,initial=None,sweeps=100,tolerance=1e-10,order=None):
    # Both branches validate independently; reject combined work before running either.
    h,e=compile_model(payload);n=len(h)
    if type(sweeps) is not int or not 1<=sweeps<=1000:raise ValueError('sweeps must be integer 1..1000')
    if sweeps*n*(n+len(e))>500000:raise ValueError('mean-field work exceeds 500000')
    if (1<<n)*(n+len(e)+n*n)>786432:raise ValueError('exact enumeration work exceeds 786432')
    fitted=mean_field(payload,initial=initial,sweeps=sweeps,tolerance=tolerance,order=order)
    truth=exact(payload);m=fitted['means'];terms=[];joint=[]
    for row in truth['states']:
        factors=[-math.inf if t*z==-1 else math.log1p(t*z)-math.log(2) for t,z in zip(m,row['state'])]
        logq=math.fsum(factors) if all(math.isfinite(v) for v in factors) else None
        q=math.exp(logq) if logq is not None else 0.0
        if q>0:terms.append(q*(logq-row['log_probability']))
        joint.append({'state':row['state'][:],'q_probability':q,'q_log_probability':logq})
    return {'exact':truth,'mean_field':fitted,'q_states':joint,
            'reverse_kl_direct':math.fsum(terms),'log_partition_minus_elbo':truth['log_partition']-fitted['elbo'],
            'max_mean_error':max(abs(a-b) for a,b in zip(truth['means'],m)),
            'scope':'reverse KL is KL(q || p), not KL(p || q); zero product masses have null log probability; all reported arithmetic is floating point'}
