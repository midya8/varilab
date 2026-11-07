import copy,importlib.util,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from varilab.curvature import curvature_demo
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_curvature',ROOT/'examples/render_curvature.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
NS={'s':'http://www.w3.org/2000/svg'}
class FigureTests(unittest.TestCase):
    def test_exact_regeneration(self):self.assertEqual((ROOT/'assets/curvature-audit.svg').read_text(),mod.render(curvature_demo()))
    def test_complete_metadata(self):
        root=ET.fromstring(mod.render(curvature_demo()));self.assertEqual(json.loads(root.find('s:metadata',NS).text),curvature_demo())
    def test_all_inertia_count_coordinates(self):
        p=curvature_demo();root=ET.fromstring(mod.render(p));bars=[e for e in root.findall('s:rect',NS) if 'data-row' in e.attrib];self.assertEqual(len(bars),15)
        for e in bars:
            r=int(e.attrib['data-row']);key=e.attrib['data-sign'];v=p['rows'][r]['audit']['inertia'][key]
            self.assertEqual(int(e.attrib['data-value']),v);self.assertEqual(int(e.attrib['width']),45*v)
            self.assertEqual(int(e.attrib['x']),595+150*['positive','negative','zero'].index(key));self.assertEqual(int(e.attrib['y']),177+75*r)
    def test_reject_changed_fixture(self):
        p=curvature_demo();p['rows'][0]['audit']['inertia']['positive']=1
        with self.assertRaises(ValueError):mod.render(p)
