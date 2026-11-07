import copy,html,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from fractions import Fraction as F
from examples.render_map import render
from varilab.decoding import map_demo
NS={'s':'http://www.w3.org/2000/svg'}


class MapFigureTests(unittest.TestCase):
    def test_score_geometry_common_zero(self):
        root=ET.fromstring(render(map_demo()));rows=[r for r in root.findall('s:rect',NS) if 'data-method' in r.attrib];self.assertEqual(len(rows),2)
        for row,score in zip(rows,(F(-385,256),F(513,256))):
            self.assertEqual(F(row.attrib['data-score']),score);self.assertEqual(float(row.attrib['y']),min(310,310-50*float(score)));self.assertEqual(float(row.attrib['height']),50*abs(float(score)))
    def test_exact_certificates_in_metadata(self):
        d=map_demo();root=ET.fromstring(render(d));meta=json.loads(root.find('s:metadata',NS).text)
        for key in ('unique_case','tied_case','marginal_modes','marginal_mode_score','score_gap','gates','selection','scope','independent_128'):self.assertEqual(meta[key],d[key])
        self.assertEqual(meta['display_marginal_probabilities'],[[round(v,12) for v in row] for row in d['marginal_probabilities']])
    def test_reproducible(self):self.assertEqual(Path('assets/map-audit.svg').read_text(),render(map_demo()))
    def test_changed_inputs_rejected(self):
        d=copy.deepcopy(map_demo());d['unique_case']['maximizing_count']=2
        with self.assertRaises(ValueError):render(d)
