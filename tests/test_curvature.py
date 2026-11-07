import copy,itertools,json,math,os,subprocess,sys,unittest
from fractions import Fraction as F
from unittest.mock import patch
from varilab import curvature,mean_field,elbo,elbo_gradient
from varilab.curvature import _inertia


def unpack(a):return [[F(v['numerator'],v['denominator']) for v in r] for r in a]
def mul(a,b):return [[sum(x*y for x,y in zip(r,c)) for c in zip(*b)] for r in a]
def transpose(a):return list(map(list,zip(*a)))
def det(a):
    if not a:return F(1)
    return sum((-1)**j*a[0][j]*det([r[:j]+r[j+1:] for r in a[1:]]) for j in range(len(a)))
def model(j=3):return {'version':1,'fields':[0,0],'edges':[{'i':0,'j':1,'weight':j}]}


class CurvatureTests(unittest.TestCase):
    def check_certificate(self,out):
        a,t,d=map(unpack,(out['hessian'],out['congruence'],out['block_diagonal']))
        self.assertEqual(mul(mul(transpose(t),a),t),d);self.assertNotEqual(det(t),0)
        self.assertEqual(sum(out['inertia'].values()),len(a))
        for sign,w in out['curvature_witnesses'].items():
            v=[F(x['numerator'],x['denominator']) for x in w['direction']]
            q=sum(v[i]*a[i][j]*v[j] for i in range(len(a)) for j in range(len(a)))
            self.assertEqual(q,F(w['quadratic_form']['numerator'],w['quadratic_form']['denominator']))
            self.assertGreater(q if sign=='positive' else -q,0)
    def test_zero_start_is_saddle(self):
        out=curvature(model(),[0,0]);self.check_certificate(out)
        self.assertEqual(out['inertia'],{'positive':1,'negative':1,'zero':0})
        self.assertEqual(out['classification'],'indefinite')
        self.assertEqual(out['gradient_float'],[0,0])
        self.assertGreater(elbo(model(),[.001,.001])['elbo'],elbo(model(),[0,0])['elbo'])
    def test_broken_symmetry_negative_definite(self):
        fit=mean_field(model(),initial=[.1,.1],tolerance=1e-12)
        out=curvature(model(),fit['means']);self.check_certificate(out)
        self.assertEqual(out['inertia'],{'positive':0,'negative':2,'zero':0})
    def test_critical_singular(self):
        out=curvature(model(1),[0,0]);self.check_certificate(out)
        self.assertEqual(out['classification'],'negative_semidefinite');self.assertEqual(out['inertia']['zero'],1)
    def test_one_bit_boundary_no_tolerance(self):
        for j,expected in [(math.nextafter(1,0),'negative_definite'),(1,'negative_semidefinite'),(math.nextafter(1,2),'indefinite')]:
            out=curvature(model(j),[0,0]);self.check_certificate(out);self.assertEqual(out['classification'],expected)
    def test_independent_formula(self):
        p={'version':1,'fields':[.1,.2,-.3],'edges':[]};m=[.2,-.5,.75]
        out=curvature(p,m);self.check_certificate(out)
        a=unpack(out['hessian'])
        for i in range(3):self.assertEqual(a[i][i],-1/(1-F(m[i])**2))
        self.assertEqual(out['classification'],'negative_definite')
    def test_hessian_gradient_finite_differences(self):
        p={'version':1,'fields':[.1,.2,-.3],'edges':[{'i':0,'j':2,'weight':.7}]};m=[.2,-.5,.75];a=unpack(curvature(p,m)['hessian'])
        for j in range(3):
            plus=m[:];minus=m[:];plus[j]+=1e-6;minus[j]-=1e-6
            gp=elbo_gradient(p,plus);gm=elbo_gradient(p,minus)
            for i in range(3):self.assertAlmostEqual((gp[i]-gm[i])/2e-6,float(a[i][j]),places=7)
    def test_729_symmetric_integer_certificates(self):
        for values in itertools.product((-1,0,1),repeat=6):
            a=[[F(0) for _ in range(3)] for _ in range(3)]
            for v,(i,j) in zip(values,[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):a[i][j]=a[j][i]=F(v)
            d,t,c,b,w=_inertia(a)
            self.assertEqual(mul(mul(transpose(t),a),t),d);self.assertNotEqual(det(t),0)
            self.assertEqual(sum(c.values()),3)
            # Independent characteristic-polynomial oracle, no elimination.
            trace=sum(a[i][i] for i in range(3))
            e2=sum(a[i][i]*a[j][j]-a[i][j]**2 for i in range(3) for j in range(i+1,3))
            determinant=det(a)
            if determinant:
                positive_definite=all(det([r[:k] for r in a[:k]])>0 for k in (1,2,3))
                negative_definite=all((-1)**k*det([r[:k] for r in a[:k]])>0 for k in (1,2,3))
                expected=(3,0,0) if positive_definite else (0,3,0) if negative_definite else (1,2,0) if determinant>0 else (2,1,0)
            elif e2:
                expected=(1,1,1) if e2<0 else (2,0,1) if trace>0 else (0,2,1)
            elif trace:expected=(1,0,2) if trace>0 else (0,1,2)
            else:expected=(0,0,3)
            self.assertEqual(tuple(c[x] for x in ('positive','negative','zero')),expected)
            # Independent Sylvester test for strict definiteness.
            minors=[det([r[:k] for r in a[:k]]) for k in (1,2,3)]
            self.assertEqual(c['positive']==3,all(v>0 for v in minors))
            self.assertEqual(c['negative']==3,all((-1)**k*v>0 for k,v in enumerate(minors,1)))
            if det(a):self.assertEqual((-1)**c['negative'],1 if det(a)>0 else -1)
    def test_zero_diagonal_two_pivot(self):
        a=[[F(0),F(2),F(3)],[F(2),F(0),F(4)],[F(3),F(4),F(0)]]
        d,t,c,b,w=_inertia(a);self.assertEqual(b[0]['size'],2)
        self.assertEqual(c,{'positive':1,'negative':2,'zero':0})
        self.assertEqual(mul(mul(transpose(t),a),t),d)
    def test_permutation_invariance(self):
        p={'version':1,'fields':[0,0,0],'edges':[{'i':0,'j':1,'weight':2},{'i':0,'j':2,'weight':-.5}]};m=[.1,.2,.3]
        ref=curvature(p,m)
        for perm in itertools.permutations(range(3)):
            edges=[]
            for e in p['edges']:
                i,j=sorted([perm.index(e['i']),perm.index(e['j'])]);edges.append({'i':i,'j':j,'weight':e['weight']})
            out=curvature({'version':1,'fields':[0]*3,'edges':edges},[m[i] for i in perm]);self.check_certificate(out)
            self.assertEqual(ref['inertia'],out['inertia'])
    def test_near_endpoint_exact_rational(self):
        m=math.nextafter(1,0);out=curvature(model(.5),[m,-m]);self.check_certificate(out)
        self.assertEqual(unpack(out['hessian'])[0][0],-1/(1-F(m)**2));json.dumps(out,allow_nan=False)
    def test_full_twelve_dimension(self):
        p={'version':1,'fields':[0]*12,'edges':[{'i':i,'j':j,'weight':.1} for i in range(12) for j in range(i+1,12)]}
        out=curvature(p,[.25]*12);a,t,d=map(unpack,(out['hessian'],out['congruence'],out['block_diagonal']))
        self.assertEqual(mul(mul(transpose(t),a),t),d);self.assertEqual(out['inertia']['positive'],1)
    def test_no_enumeration_or_fit(self):
        with patch('varilab.core.exact',side_effect=AssertionError),patch('varilab.core.mean_field',side_effect=AssertionError):curvature(model(),[0,0])
    def test_input_isolation(self):
        p=model();m=[.1,.2];before=copy.deepcopy((p,m));out=curvature(p,m)
        out['model']['fields'][0]=9;out['means'][0]=9;self.assertEqual((p,m),before)
    def test_rejects_endpoints_invalid_means(self):
        for m in ([1,0],[-1,0],[True,0],[float('nan'),0],[0],None,[2,0]):
            with self.assertRaises(ValueError):curvature(model(),m)
    def test_rejects_invalid_model(self):
        for p in ({},dict(model(),version=True),dict(model(),extra=0)):
            with self.assertRaises(ValueError):curvature(p,[0,0])
    def test_fields_do_not_change_curvature(self):
        p=model();out=curvature(p,[.2,.3]);p['fields']=[4,-7];new=curvature(p,[.2,.3])
        self.assertEqual(out['hessian'],new['hessian']);self.assertNotEqual(out['gradient_float'],new['gradient_float'])
    def test_cli_matches_api(self):
        run=subprocess.run([sys.executable,'-m','varilab','curvature','examples/coupled.json','--means','[0,0]'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout),curvature(model(),[0,0]))
    def test_cli_rejects_endpoint(self):
        run=subprocess.run([sys.executable,'-m','varilab','curvature','examples/coupled.json','--means','[1,0]'],capture_output=True,text=True)
        self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'');self.assertIn('interior',run.stderr)

    def test_zero_matrix_and_rank_one(self):
        for a,expected in [([[F(0)]*3 for _ in range(3)],(0,0,3)),([[F(1)]*3 for _ in range(3)],(1,0,2)),([[-F(1)]*3 for _ in range(3)],(0,1,2))]:
            d,t,c,b,w=_inertia(a);self.assertEqual(mul(mul(transpose(t),a),t),d)
            self.assertEqual(tuple(c[x] for x in ('positive','negative','zero')),expected)
    def test_deterministic_certificates(self):
        self.assertEqual(curvature(model(-3),[.13,-.25]),curvature(model(-3),[.13,-.25]))
