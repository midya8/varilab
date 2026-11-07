"""Directly specified independent and symmetric two-mode fixtures."""
from .core import compare


def demo():
    independent={'version':1,'fields':[.4,-.7,.2],'edges':[]}
    coupled={'version':1,'fields':[0,0],'edges':[{'i':0,'j':1,'weight':3}]}
    arms=[('symmetric',[0,0]),('positive',[.1,.1]),('negative',[-.1,-.1])]
    results=[{'name':name,'comparison':compare(coupled,initial=initial,sweeps=100,tolerance=1e-12)} for name,initial in arms]
    control=compare(independent,sweeps=10,tolerance=1e-12)
    zero,positive,negative=[r['comparison'] for r in results]
    assert control['max_mean_error']<1e-12 and abs(control['reverse_kl_direct'])<1e-12
    assert positive['mean_field']['elbo']>zero['mean_field']['elbo']+1
    assert positive['reverse_kl_direct']<zero['reverse_kl_direct']-1
    assert positive['max_mean_error']>.9 and zero['max_mean_error']<1e-12
    assert abs(positive['mean_field']['elbo']-negative['mean_field']['elbo'])<1e-12
    assert all(r['comparison']['mean_field']['stopping_rule_met'] for r in results)
    return {'independent_control':control,'coupled_arms':results,
            'gates':{'independent_mean_error_le_1e_6':control['max_mean_error']<=1e-6,
                     'positive_mean_error_le_1e_6':positive['max_mean_error']<=1e-6,
                     'positive_elbo_gt_symmetric':positive['mean_field']['elbo']>zero['mean_field']['elbo'],
                     'positive_reverse_kl_lt_symmetric':positive['reverse_kl_direct']<zero['reverse_kl_direct']},
            'selection':'first directly specified independent fields, coupling J=3 and symmetric/+0.1/-0.1 starts; no parameter, seed or checkpoint search',
            'scope':'a constructed objective-versus-marginal counterexample, not a ranking of inference methods or empirical performance estimate'}
