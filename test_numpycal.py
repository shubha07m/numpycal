"""Run offline: python -m unittest -v"""
import json
import ast
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import drills
import examples
from app import app
from engine import calculate, parse_array, OPERATIONS, packed


class Algorithms(unittest.TestCase):
    def test_standardize_constant_and_axes(self):
        A=np.array([[1.,5],[2,5],[3,5]])
        actual=drills.standardize(A)
        assert_allclose(actual.mean(0),0,atol=1e-15)
        assert_allclose(actual.std(0),[1,0])
        assert_allclose(drills.standardize(A,1).mean(1),0,atol=1e-15)

    def test_softmax_large_logits_and_shift_invariance(self):
        A=np.array([[1000.,1001,1002],[-1000,-1000,-1000]])
        p=drills.softmax(A)
        assert_allclose(p.sum(-1),1)
        assert_allclose(p,drills.softmax(A+12345))
        assert_allclose(p[1],[1/3]*3)

    def test_nearest_center_against_loop_oracle(self):
        rng=np.random.default_rng(42); A=rng.normal(size=(7,4)); B=rng.normal(size=(3,4))
        oracle=np.array([np.argmin([sum((a-b)**2) for b in B]) for a in A])
        assert_array_equal(drills.nearest_center(A,B),oracle)
        assert_array_equal(drills.nearest_center(np.array([[0,0]]),np.array([[1,0],[-1,0]])),[0])

    def test_least_squares_singular_and_intercept(self):
        A=np.array([[1,0],[1,1],[1,2],[1,3]])
        assert_allclose(drills.least_squares(A,np.array([1,3,5,7])),[1,2])
        A=np.ones((3,2)); w=drills.least_squares(A,np.array([2,2,2]))
        assert_allclose(A@w,[2,2,2])

    def test_convolution_against_loop_oracle(self):
        rng=np.random.default_rng(11)
        for shape in [(3,4),(5,5)]:
            A=rng.normal(size=shape); B=rng.normal(size=(2,3))
            for pad in [0,1,2]:
                for stride in [1,2,3]:
                    P=np.pad(A,pad)
                    out=np.array([[np.sum(P[i:i+2,j:j+3]*B) for j in range(0,P.shape[1]-2,stride)] for i in range(0,P.shape[0]-1,stride)])
                    assert_allclose(drills.convolution(A,B,pad,stride),out)

    def test_gradient_descent_and_weighted_cost(self):
        A=np.array([[1.,0],[1,1],[1,2],[1,3]])
        assert_allclose(examples.linear_regression(A,np.array([1,3,5,7])).ravel(),[1,2],atol=1e-5)
        A=np.array([[1.,-2],[1,-1],[1,1],[1,2]]); y=np.array([0,0,1,1])
        w,cost=examples.logistic_regression(A,y)
        self.assertLess(cost,.03)
        assert_array_equal((A@w).ravel()>0,y.astype(bool))
        logits=A@w; weights=np.where(y[:,None]==1,2,1)
        oracle=np.sum(weights*(np.logaddexp(0,logits)-y[:,None]*logits))/weights.sum()
        assert_allclose(cost,oracle)

    def test_gini_and_stump(self):
        self.assertEqual(examples.gini(np.array([0,0,1,1])),.5)
        self.assertEqual(examples.gini(np.array([1,1])),0)
        assert_allclose(examples.decision_stump(np.array([[0],[1],[3],[4]]),np.array([0,0,1,1])),[0,2,0])
        with self.assertRaises(ValueError): examples.decision_stump(np.ones((3,1)),np.array([0,1,1]))

    def test_knn_and_cross_entropy(self):
        assert_array_equal(examples.knn(np.array([[0,0],[9,9]]),np.array([[0,0],[1,0],[0,1],[8,9],[9,8],[9,9]]),np.array([0,0,0,1,1,1])),[0,1])
        assert_allclose(examples.cross_entropy(np.array([[1000.,-1000.]]),np.array([1])),2000)
        assert_allclose(examples.cross_entropy(np.zeros((2,3)),np.array([0,2])),np.log(3))


class Bench(unittest.TestCase):
    def setUp(self):
        self.A=np.array([[1,2,3],[4,5,6],[7,8,10]])
        self.B=np.array([[1,2,3],[4,5,6],[7,8,9]])
        self.client=app.test_client()

    def test_parser_literals_modes_and_rejections(self):
        assert_array_equal(parse_array('1,2;3,4','2'),[[1,2],[3,4]])
        assert_array_equal(parse_array('[[1,2],[3,4]]','1'),[1,2,3,4])
        assert_array_equal(parse_array('1 2 3','2'),[[1,2,3]])
        for bad in ['', '1,2\n3', '[NaN]', '[1e999]', '[True]', '["a"]', '[]', '__import__("os").system("ls")']:
            with self.subTest(bad=bad),self.assertRaises(ValueError): parse_array(bad)

    def test_round_trip_arbitrary_result_dimensions(self):
        for A in [np.array(3.),np.array([1,2]),np.ones((2,1,3)),np.ones((1,2,1,2))]:
            r=packed(A); back=parse_array(r['literal'],'auto')
            self.assertEqual(back.shape,A.shape)
            assert_allclose(back,A)

    def test_acceptance_operations(self):
        assert_array_equal(calculate('transpose',self.A)['data'],self.A.T)
        assert_allclose(calculate('matmul',self.A,self.B)['data'],self.A@self.B)
        assert_allclose(calculate('softmax',self.A,params={'axis':'-1'})['data'],drills.softmax(self.A))
        r=calculate('mean',self.A,params={'axis':'0','keepdims':True})
        self.assertEqual(r['shape'],'(1, 3)')
        self.assertIn('np.mean(A, axis=0, keepdims=True)',r['expression'])

    def test_every_deck_button(self):
        for op in OPERATIONS:
            with self.subTest(op=op):
                params={'axis':'0','reshape_to':'1,9','padding':0,'stride':1}
                A=self.A if op!='squeeze' else np.ones((1,3))
                B=np.array([1,2,3]) if op=='least_squares' else self.B
                r=calculate(op,A,B if OPERATIONS[op]['binary'] else None,params)
                self.assertIn('dtype',r); self.assertIn('expression',r)

    def test_error_cases_do_not_crash_server(self):
        cases=[dict(operation='inverse',A='[[1,1],[1,1]]'),dict(operation='matmul',A='[[1,2,3]]',B='[[1,2]]'),dict(operation='reshape',A='1,2,3',params={'reshape_to':'2,2'}),dict(operation='clip',A='1',params={'clip_min':9,'clip_max':2}),dict(operation='expand_dims',A='1',params={'axis':'None'}),dict(operation='convolution',A='[[1]]',B='[[1,2],[3,4]]'),dict(operation='divide',A='1',B='0'),dict(operation='exp',A='1000'),dict(operation='log',A='-1')]
        for payload in cases:
            with self.subTest(payload=payload):
                response=self.client.post('/api/run',json=payload)
                self.assertEqual(response.status_code,400); self.assertIn('error',response.json)
        self.assertEqual(self.client.get('/').status_code,200)

    def test_all_patterns_and_examples(self):
        self.assertEqual(len(drills.PATTERNS),5)
        for d in drills.PATTERNS:
            response=self.client.post('/api/pattern',json={'id':d['id']})
            self.assertEqual(response.status_code,200,response.json)
            self.assertTrue(response.json['steps'])
        for e in examples.EXAMPLES:
            response=self.client.post('/api/example',json={'id':e['id']})
            self.assertEqual(response.status_code,200,response.json)

    def test_displayed_expressions_reproduce_results(self):
        # Only execute trusted app-generated reference code, never user inputs.
        for d in drills.PATTERNS:
            r=self.client.post('/api/pattern',json={'id':d['id']}).json
            namespace={'A':np.array(d['A']), 'B':np.array(d['B'])}
            exec(r['expression'],namespace)
            assert_allclose(namespace['result'],r['data'])
        for e in examples.EXAMPLES:
            r=self.client.post('/api/example',json={'id':e['id']}).json
            namespace={'A':np.array(e['A'],dtype=float),'B':np.array(e['B'],dtype=float)}
            exec(r['expression'],namespace)
            assert_allclose(namespace['result'],r['data'])
        for op in OPERATIONS:
            with self.subTest(op=op):
                A=self.A if op!='squeeze' else np.ones((1,3))
                B=np.array([1,2,3]) if op=='least_squares' else self.B
                r=calculate(op,A,B if OPERATIONS[op]['binary'] else None,{'axis':'0','reshape_to':'1,9'})
                tree=ast.parse(r['expression'])
                if isinstance(tree.body[-1],ast.Expr):
                    tree.body[-1]=ast.Assign(targets=[ast.Name(id='result',ctx=ast.Store())],value=tree.body[-1].value)
                    ast.fix_missing_locations(tree)
                namespace={'A':A.copy(),'B':B.copy()}
                exec(compile(tree,'<reference>','exec'),namespace)
                assert_allclose(namespace['result'],r['data'])
    def test_invalid_json_and_unknown_operation(self):
        for path in ['/api/run','/api/parse','/api/pattern','/api/example']:
            self.assertEqual(self.client.post(path,json=[]).status_code,400)
        self.assertEqual(self.client.post('/api/run',json={'operation':'eval'}).status_code,400)
        self.assertEqual(self.client.post('/api/parse',data='x'*300000,content_type='application/json').status_code,413)

    def test_no_remote_assets_or_fetch(self):
        html=self.client.get('/').text
        self.assertNotIn('fetch(',html)
        self.assertNotIn('src="http',html)
        self.assertNotIn('href="http',html)
        self.assertIn("connect-src 'self'",self.client.get('/').headers['Content-Security-Policy'])




class ShapePreview(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_preview_rules_match_actual_shapes(self):
        from shapes import predict_shape
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        for op in OPERATIONS:
            with self.subTest(op=op):
                a = np.ones((1, 3)) if op == 'squeeze' else A
                b = np.array([1, 2, 3]) if op == 'least_squares' else A
                params = dict(axis='0', reshape_to='1,9', padding=1, stride=2)
                prediction = predict_shape(op, a, b, params)
                result = calculate(op, a, b if OPERATIONS[op]['binary'] else None, params)
                if op == 'unique':
                    self.assertIsNone(prediction['shape'])
                else:
                    self.assertEqual(prediction['shape'], result['dimensions'])
                self.assertTrue(result['explanation'])

    def test_broadcasting_matmul_and_reduction_shapes(self):
        from shapes import predict_shape
        for a, b in [((3, 3), (3, 1)), ((2, 1, 3), (4, 3)), ((), (3,)), ((3,), (3, 1))]:
            with self.subTest(a=a, b=b):
                p = predict_shape('add', np.ones(a), np.ones(b))
                self.assertEqual(p['shape'], list((np.ones(a) + np.ones(b)).shape))
                self.assertIn('via broadcasting', p['hint'])
        for a, b in [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 2)), ((2, 3), (3, 4))]:
            self.assertEqual(predict_shape('matmul', np.ones(a), np.ones(b))['shape'], list((np.ones(a) @ np.ones(b)).shape))
        for axis in ['None', '0', '1', '-1', '-2,-1']:
            for keep in [False, True]:
                params = dict(axis=axis, keepdims=keep)
                p = predict_shape('sum', np.ones((2, 3, 4)), params=params)
                self.assertEqual(p['shape'], calculate('sum', np.ones((2, 3, 4)), params=params)['dimensions'])

    def test_previews_do_not_calculate(self):
        from unittest.mock import patch
        with patch('app.calculate', side_effect=AssertionError('preview called calculation')):
            response = self.client.post('/api/predict', json=dict(operation='inverse', A='[[1,1],[1,1]]'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['shape'], [2, 2])  # Shape is valid even if the matrix is singular.

    def test_friendly_alignment_errors(self):
        for path in ['/api/run', '/api/predict']:
            response = self.client.post(path, json=dict(operation='add', A='[[1,2,3],[4,5,6],[7,8,9]]', B='[[1,2,3],[4,5,6]]'))
            self.assertEqual(response.status_code, 400)
            self.assertIn("Shapes (3,3) and (2,3) don't align", response.json['error'])
            self.assertIn('transpose or reshape', response.json['error'])
        response = self.client.post('/api/predict', json=dict(operation='matmul', A='[[1,2,3]]', B='[[1,2]]'))
        self.assertEqual(response.status_code, 400)
        self.assertIn('inner', response.json['error'].replace('last axis of A must match the first axis of B', 'inner'))

    def test_python_copies_are_standalone(self):
        from contextlib import redirect_stdout
        from io import StringIO
        cases = [('/api/pattern', dict(id=p['id'])) for p in drills.PATTERNS]
        cases += [('/api/example', dict(id=e['id'])) for e in examples.EXAMPLES]
        cases += [('/api/run', dict(operation='transpose', A='[[1,2],[3,4]]'))]
        for path, payload in cases:
            with self.subTest(path=path, payload=payload):
                result = self.client.post(path, json=payload).json
                namespace = {}
                with redirect_stdout(StringIO()):
                    exec(result['python_code'], namespace)
                assert_allclose(namespace['result'], result['data'])
                self.assertIn('# shape ' + result['shape'], result['python_code'])
                assert_allclose(json.loads(result['literal']), result['data'])


if __name__ == "__main__": unittest.main()
