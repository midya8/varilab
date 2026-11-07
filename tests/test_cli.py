import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from varilab import exact,mean_field,compare
from varilab.demo import demo


def run(*args):return subprocess.run([sys.executable,'-m','varilab',*args],capture_output=True,text=True)


class CLITests(unittest.TestCase):
    def test_demo_matches_api(self):
        out=run('demo');self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),demo())

    def test_exact_fit_compare_match_api(self):
        m={'version':1,'fields':[.2,-.3],'edges':[{'i':0,'j':1,'weight':.7}]}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'m.json');p.write_text(json.dumps(m))
            for cmd,fn in [('exact',exact),('fit',mean_field),('compare',compare)]:
                kw={} if cmd=='exact' else {'initial':[.1,-.2],'order':[1,0],'sweeps':3,'tolerance':0}
                args=[] if cmd=='exact' else ['--initial','[0.1,-0.2]','--order','[1,0]','--sweeps','3','--tolerance','0']
                out=run(cmd,str(p),*args);self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),fn(m,**kw))

    def test_malformed_duplicate_nonfinite_oversize(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'m')
            for body in ('{','{"version":1,"version":1,"fields":[0],"edges":[]}','{"version":1,"fields":[NaN],"edges":[]}',' '*262145):
                p.write_text(body);out=run('exact',str(p));self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertNotIn('Traceback',out.stderr)

    def test_invalid_flags_and_settings(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'m');p.write_text('{"version":1,"fields":[0],"edges":[]}')
            for args in (['--initial','[NaN]'],['--initial','[true]'],['--initial','{'],['--order','[true]'],['--sweeps','0'],['--tolerance','nan'],['--tolerance','-1'],['--unknown']):
                out=run('fit',str(p),*args);self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertNotIn('Traceback',out.stderr)

    def test_missing_file_unknown_command_help(self):
        out=run('exact','/not/a/model');self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'')
        self.assertEqual(run('unknown').returncode,2);self.assertEqual(run('--help').returncode,0)
        self.assertIn('compare',run('--help').stdout)

    def test_boundary_point_mass_is_json_finite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'m');p.write_text('{"version":1,"fields":[100],"edges":[]}')
            out=run('compare',str(p));self.assertEqual(out.returncode,0,out.stderr);self.assertNotIn('Infinity',out.stdout);self.assertNotIn('NaN',out.stdout)
            q=json.loads(out.stdout)['q_states'];self.assertIsNone(q[0]['q_log_probability']);self.assertEqual(q[1]['q_probability'],1)
