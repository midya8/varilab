import copy
import itertools
import math
import random
import unittest
from unittest.mock import patch
from varilab import exact,mean_field,compare,elbo,elbo_gradient
from varilab.demo import demo


def model(h,edges=()):return {'version':1,'fields':list(h),'edges':[{'i':i,'j':j,'weight':w} for i,j,w in edges]}
def close(test,a,b,tol=1e-11):test.assertAlmostEqual(a,b,delta=tol)


class CoreTests(unittest.TestCase):
    def test_independent_analytic_partition_moments(self):
        h=[.2,-.7,0,1.3];p=exact(model(h));close(self,p['log_partition'],sum(math.log(2*math.cosh(v)) for v in h))
        for i,v in enumerate(h):
            close(self,p['means'][i],math.tanh(v))
            for j in range(len(h)):close(self,p['covariance'][i][j],1-math.tanh(v)**2 if i==j else 0)

    def test_two_spin_symmetric_analytic(self):
        for j in (-3,-.2,0,.5,3):
            p=exact(model([0,0],[(0,1,j)]));close(self,p['log_partition'],math.log(4*math.cosh(j)))
            close(self,p['means'][0],0);close(self,p['means'][1],0);close(self,p['covariance'][0][1],math.tanh(j))

    def test_exhaustive_direct_exponentials(self):
        checked=0
        for h in itertools.product((-.5,.5),repeat=3):
            for weights in itertools.product((-1,0,1),repeat=3):
                edges=[(0,1,weights[0]),(0,2,weights[1]),(1,2,weights[2])];p=exact(model(h,edges))
                z=list(itertools.product((-1,1),repeat=3));a=[math.exp(sum(x*y for x,y in zip(h,s))+sum(w*s[i]*s[j] for i,j,w in edges)) for s in z];total=sum(a)
                close(self,p['log_partition'],math.log(total))
                for r,w,s in zip(p['states'],a,z):self.assertEqual(r['state'],list(s));close(self,r['probability'],w/total)
                for i in range(3):close(self,p['means'][i],sum(w*s[i] for w,s in zip(a,z))/total)
                checked+=1
        self.assertEqual(checked,216)

    def test_full_joint_and_pair_moments(self):
        p=exact(model([.3,-.2,.1],[(0,1,.5),(1,2,-.7)]));close(self,sum(r['probability'] for r in p['states']),1)
        for i in range(3):
            for j in range(3):
                second=sum(r['probability']*r['state'][i]*r['state'][j] for r in p['states'])
                close(self,p['second_moments'][i][j],second);close(self,p['covariance'][i][j],second-p['means'][i]*p['means'][j])

    def test_elbo_direct_q_enumeration(self):
        m=model([.2,-.5],[(0,1,1.2)]);means=[.3,-.4];truth=exact(m);parts=[];entropy=0
        for row in truth['states']:
            q=math.prod((1+t*z)/2 for t,z in zip(means,row['state']));logpot=row['log_probability']+truth['log_partition'];parts.append(q*logpot);entropy-=q*math.log(q)
        out=elbo(m,means);close(self,out['expected_log_potential'],sum(parts));close(self,out['entropy'],entropy);close(self,out['elbo'],sum(parts)+entropy)

    def test_elbo_gradient_all_coordinates_finite_difference(self):
        m=model([.3,-.4,.2],[(0,1,.7),(1,2,-.2),(0,2,.1)]);x=[.2,-.3,.4];g=elbo_gradient(m,x)
        for i in range(3):
            a=x[:];b=x[:];a[i]+=1e-6;b[i]-=1e-6;fd=(elbo(m,a)['elbo']-elbo(m,b)['elbo'])/2e-6;close(self,g[i],fd,1e-9)
        with self.assertRaises(ValueError):elbo_gradient(m,[1,0,0])

    def test_coordinate_update_is_conditional_optimum(self):
        m=model([.2,-.3],[(0,1,1.1)]);initial=[-.7,.4];p=mean_field(m,initial=initial,sweeps=1,tolerance=0)
        first=math.tanh(.2+1.1*.4);close(self,p['history'][1]['means'][0],first);self.assertEqual(p['history'][1]['means'][1],.4)
        second=math.tanh(-.3+1.1*first);close(self,p['means'][1],second)
        optimum=p['history'][1]['elbo']
        for v in (i/100 for i in range(-100,101)):self.assertLessEqual(elbo(m,[v,.4])['elbo'],optimum+1e-12)

    def test_every_coordinate_history_monotone(self):
        for j in (-3,.2,3):
            p=mean_field(model([.1,-.2,.3],[(0,1,j),(1,2,-j)]),initial=[-.8,.6,.3],sweeps=30,tolerance=0)
            for a,b in zip(p['history'],p['history'][1:]):self.assertGreaterEqual(b['elbo']+1e-12,a['elbo'])
            self.assertEqual(len(p['history']),p['coordinate_updates']+1)
            self.assertEqual(p['coordinate_updates'],3*p['sweeps_completed'])

    def test_history_recomputed_and_snapshots_independent(self):
        m=model([.2,.1],[(0,1,.7)]);p=mean_field(m,initial=[.1,-.1],sweeps=4,tolerance=0)
        for i,r in enumerate(p['history']):self.assertEqual(i,r['update']);close(self,r['elbo'],elbo(m,r['means'])['elbo'])
        self.assertEqual(p['means'],p['history'][-1]['means']);p['history'][0]['means'][0]=99;self.assertNotEqual(p['initial'][0],99)
        p['history'][-1]['means'][0]=99;self.assertNotEqual(p['means'][0],99)

    def test_independent_cavi_exact_from_any_start(self):
        m=model([.2,-.5,0]);p=compare(m,initial=[1,-1,.7],sweeps=10)
        self.assertLess(p['max_mean_error'],1e-12);close(self,p['reverse_kl_direct'],0);self.assertTrue(p['mean_field']['stopping_rule_met'])
        for i,v in enumerate(m['fields']):close(self,p['mean_field']['means'][i],math.tanh(v))

    def test_kl_identity_independent_direct_sum(self):
        for h in itertools.product((-.4,.4),repeat=2):
            for j in (-2,0,2):
                for init in ([0,0],[.2,-.3],[1,-1]):
                    p=compare(model(h,[(0,1,j)]),initial=init,sweeps=10,tolerance=0)
                    kl=sum(q['q_probability']*math.log(q['q_probability']/r['probability']) for q,r in zip(p['q_states'],p['exact']['states']) if q['q_probability']>0)
                    close(self,p['reverse_kl_direct'],kl);close(self,p['log_partition_minus_elbo'],kl)
                    self.assertGreaterEqual(kl,-1e-12)

    def test_symmetric_fixed_point_is_not_maximum(self):
        m=model([0,0],[(0,1,3)]);p=mean_field(m)
        self.assertTrue(p['stopping_rule_met']);self.assertEqual(p['means'],[0,0]);self.assertEqual(p['max_coordinate_residual'],0)
        self.assertGreater(elbo(m,[.001,.001])['elbo'],p['elbo'])
        self.assertEqual(elbo_gradient(m,[0,0]),[0,0])

    def test_higher_elbo_worse_means_in_demo(self):
        p=demo();a,b,c=[r['comparison'] for r in p['coupled_arms']]
        self.assertGreater(b['mean_field']['elbo'],a['mean_field']['elbo']);self.assertLess(b['reverse_kl_direct'],a['reverse_kl_direct']);self.assertGreater(b['max_mean_error'],a['max_mean_error'])
        close(self,b['mean_field']['means'][0],-c['mean_field']['means'][0]);close(self,b['mean_field']['elbo'],c['mean_field']['elbo'])
        self.assertTrue(p['gates']['independent_mean_error_le_1e_6']);self.assertFalse(p['gates']['positive_mean_error_le_1e_6'])

    def test_mean_field_loses_cross_covariance(self):
        p=compare(model([0,0],[(0,1,3)]));close(self,p['exact']['covariance'][0][1],math.tanh(3));self.assertEqual(p['mean_field']['covariance'][0][1],0)

    def test_label_symmetry_fields_and_initial_sign(self):
        edges=[(0,1,.8),(1,2,-.2)];a=compare(model([.2,-.1,.3],edges),initial=[.1,.2,-.1]);b=compare(model([-.2,.1,-.3],edges),initial=[-.1,-.2,.1])
        close(self,a['exact']['log_partition'],b['exact']['log_partition']);close(self,a['mean_field']['elbo'],b['mean_field']['elbo'])
        for x,y in zip(a['mean_field']['means'],b['mean_field']['means']):close(self,x,-y)

    def test_order_and_permutation_equivariance(self):
        # Reversing coordinate labels and update order is an identical sequence of updates.
        a=mean_field(model([.2,-.3],[(0,1,.9)]),initial=[.1,-.2],order=[0,1],sweeps=3,tolerance=0)
        b=mean_field(model([-.3,.2],[(0,1,.9)]),initial=[-.2,.1],order=[1,0],sweeps=3,tolerance=0)
        for x,y in zip(a['history'],b['history']):close(self,x['elbo'],y['elbo']);self.assertEqual(x['means'],y['means'][::-1])

    def test_extreme_log_probabilities_retained(self):
        p=exact(model([100]*6,[(i,j,100) for i in range(6) for j in range(i+1,6)]))
        self.assertTrue(any(r['probability']==0 and math.isfinite(r['log_probability']) for r in p['states']))
        close(self,sum(r['probability'] for r in p['states']),1)
        out=compare(model([100,-100]));self.assertEqual(out['mean_field']['means'],[1,-1]);close(self,out['reverse_kl_direct'],out['log_partition_minus_elbo'])
        self.assertTrue(any(r['q_log_probability'] is None for r in out['q_states']))

    def test_tolerance_and_stopping_not_global_claim(self):
        p=mean_field(model([.1,.2],[(0,1,3)]),sweeps=1,tolerance=0)
        self.assertFalse(p['stopping_rule_met']);self.assertEqual(p['sweeps_completed'],1)
        m=p['means'];expected=max(abs(math.tanh(.1+3*m[1])-m[0]),abs(math.tanh(.2+3*m[0])-m[1]));close(self,p['max_coordinate_residual'],expected)

    def test_zero_edges_are_valid_and_edge_order_canonical(self):
        a=model([.1,.2,.3],[(1,2,.4),(0,1,0)]);b=model([.1,.2,.3],[(0,1,0),(1,2,.4)])
        self.assertEqual(exact(a),exact(b));self.assertEqual(mean_field(a),mean_field(b))

    def test_validation_rejects_bad_models(self):
        good=model([0,0]);bad=[None,{},dict(good,extra=1),dict(good,version=True),dict(good,fields=[]),dict(good,fields=[0]*13),dict(good,fields=[True]),dict(good,fields=[math.nan]),dict(good,fields=[101]),dict(good,edges='x')]
        for edge in ({'i':0,'j':0,'weight':1},{'i':1,'j':0,'weight':1},{'i':0,'j':2,'weight':1},{'i':False,'j':1,'weight':1},{'i':0,'j':1,'weight':math.inf},{'i':0,'j':1,'weight':1,'x':2}):bad.append(dict(good,edges=[edge]))
        bad.append(dict(good,edges=[{'i':0,'j':1,'weight':1}]*2))
        for m in bad:
            with self.subTest(m=m),self.assertRaises(ValueError):exact(m)

    def test_validation_rejects_bad_fit_settings(self):
        for kw in ({'initial':[True,0]},{'initial':[1.1,0]},{'initial':[0]},{'sweeps':True},{'sweeps':0},{'sweeps':1001},{'tolerance':-1},{'tolerance':math.inf},{'tolerance':True},{'order':[1,1]},{'order':[False,1]},{'order':[0]}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):mean_field(model([0,0]),**kw)

    def test_exact_work_preflight_and_actual_boundary(self):
        edges=[(i,j,0) for i in range(12) for j in range(i+1,12)]
        with patch('varilab.core._energy',side_effect=AssertionError('must preflight')):
            with self.assertRaisesRegex(ValueError,'786432'):exact(model([0]*12,edges[:37]))
        p=exact(model([0]*12,edges[:36]));self.assertEqual(p['state_count'],4096);close(self,p['log_partition'],12*math.log(2))

    def test_fit_work_preflight_boundary_and_compare(self):
        edges=[(i,j,0) for i in range(10) for j in range(i+1,10)]
        p=mean_field(model([0]*10,edges[:40]),sweeps=1000);self.assertTrue(p['stopping_rule_met'])
        with patch('varilab.core._elbo',side_effect=AssertionError('must preflight')):
            for fn in (mean_field,compare):
                with self.assertRaisesRegex(ValueError,'500000'):fn(model([0]*10,edges[:41]),sweeps=1000)

    def test_input_return_and_rng_isolation(self):
        m=model([.1,-.2],[(0,1,.3)]);start=[.1,.2];order=[1,0];before=copy.deepcopy((m,start,order));rng=random.getstate();p=compare(m,initial=start,order=order)
        self.assertEqual((m,start,order),before);self.assertEqual(random.getstate(),rng)
        p['mean_field']['model']['fields'][0]=99;self.assertNotEqual(p['exact']['model']['fields'][0],99)
        p['q_states'][0]['state'][0]=99;self.assertNotEqual(p['exact']['states'][0]['state'][0],99)
