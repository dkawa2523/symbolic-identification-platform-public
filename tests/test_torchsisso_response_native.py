"""Actual pinned native constrained-SIS paths, not mock regression."""
import unittest
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sympy = pytest.importorskip("sympy")
from importlib import import_module
from ml_platform_symbolic.engines.torchsisso_response import ResponseSearch
from ml_platform_symbolic.engines.torchsisso_budget import NodeBudget
from ml_platform_symbolic.response_search import compile_response_constraints,response_feasible
import types
pytestmark = pytest.mark.native

def compiled(points,lo,hi):
    variables=[dict(id=c,column=c,unit='1',domain=[None,None]) for c in ['x','z']]
    request=types.SimpleNamespace(X=np.array([[-1.,-1.],[1.,1.]]),y=np.array([-1.,1.]),variables=variables,target={'domain':[None,None]},limits={'probes':[dict(zip(['x','z'],p)) for p in points],'response_bounds':[lo,hi]})
    return compile_response_constraints(request)

def native(filename,x,y,names,response,dimension=2,budget=9):
    module=import_module('TorchSisso.'+filename.removesuffix('.py'))
    kw=dict(dimension=dimension,sis_features=2,node_budget=NodeBudget(budget,lambda name:1),response_search=response)
    if filename=='Regressor_dimension.py':
        # Native dimension filtering before response names are resolved.
        x=np.column_stack([np.cos(x[:,0]),x]);names=['excluded']+names
        kw.update(dimensionality=[sympy.Symbol('L')]+[sympy.Symbol('1')]*len(names[1:]),output_dim=sympy.Symbol('1'))
    return module.Regressor(torch.tensor(x,dtype=torch.float64),torch.tensor(y,dtype=torch.float64),names,**kw)

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(17);self.x=rng.uniform(-2,2,120);self.z=self.x+.2*rng.normal(size=120)
        self.X=np.column_stack([self.x,self.z]);self.y=self.x.copy()
    def test_reverses_native_one_term_winner_and_residual(self):
        for file in ['Regressor.py','Regressor_dimension.py']:
            cp=compiled([[-1,0],[1,0]],-1e-8,1e-8)
            def evaluate(names):return np.asarray(cp['points'])[:,[['x','z'].index(n) for n in names]]
            r=native(file,self.X,self.y,['x','z'],ResponseSearch(cp,evaluate),dimension=2)
            r.regressor_fit()
            first=r.portable_models_[0]
            self.assertEqual(first['indices'],[1])  # native unconstrained winner was x/index0
            self.assertLess(first['mse'],.1)
            for m in r.portable_models_:
                pred=m['intercept']+np.asarray(cp['points'])[:,m['indices']]@m['coefficients']
                self.assertTrue(response_feasible(pred,cp))
                self.assertTrue(r.node_budget.admissible(m['indices'],r.node_costs))
            last=r.portable_models_[-1]
            np.testing.assert_allclose(r.residual.numpy().ravel(),self.y-(last['intercept']+self.X[:,last['indices']]@last['coefficients']))
    def test_intercept_is_free_and_standardization_preserves_prediction(self):
        for file in ['Regressor.py','Regressor_dimension.py']:
            X=self.X*np.array([3,7])+[10,-6]; y=2*X[:,0]+4
            cp=compiled([[10,-6]],30,31)
            r=native(file,X,y,['x','z'],ResponseSearch(cp,lambda names:np.asarray(cp['points'])),dimension=1)
            r.regressor_fit();m=r.portable_models_[0]
            self.assertTrue(response_feasible(m['intercept']+np.asarray(cp['points'])[:,m['indices']]@m['coefficients'],cp))
            self.assertGreater(m['mse'],1) # constrained optimum isn't unconstrained ymean center fit
            self.assertNotAlmostEqual(m['intercept'],4,places=2)
    def test_no_feasible_singleton_does_not_stop_two_terms(self):
        # f(0,0)=0, f(1,0)=1, f(0,1)=1 requires two descriptors.
        cp=compiled([[0,0],[1,0],[0,1]],None,None)
        for t,v in zip(cp['terms'],[0,1,1]):t.update(lower=v-1e-7,upper=v+1e-7)
        for file in ['Regressor.py','Regressor_dimension.py']:
            r=native(file,self.X,self.x+self.z,['x','z'],ResponseSearch(cp,lambda names:np.asarray(cp['points'])))
            r.regressor_fit();self.assertEqual(len(r.portable_models_),1);self.assertEqual(r.portable_models_[0]['term_count'],2)
            self.assertTrue(r.response_summary_['subset_counts'].get('infeasible',0)>0)
    def test_all_infeasible_fails_closed(self):
        cp=compiled([[0,0],[1,0],[0,1]],None,None)
        for t,v in zip(cp['terms'],[0,1,1]):t.update(lower=v-1e-5,upper=v+1e-5)
        for file in ['Regressor.py','Regressor_dimension.py']:
            r=native(file,self.X,self.y,['x','z'],ResponseSearch(cp,lambda names:np.asarray(cp['points'])),dimension=1)
            with self.assertRaisesRegex(ValueError,'No feasible'):r.regressor_fit()
    def test_rank_deficient_and_nonfinite_candidates(self):
        cp=compiled([[0,0]],-1,1)
        X=np.column_stack([self.x,self.x]);cp['points']=[[0,0]]
        r=native('Regressor.py',X,self.x,['x','z'],ResponseSearch(cp,lambda names:np.zeros((1,2))))
        r.regressor_fit();self.assertTrue(np.isfinite(r.portable_models_[-1]['mse']))
        r=native('Regressor.py',self.X,self.y,['x','z'],ResponseSearch(cp,lambda names:np.array([[np.nan,0.]])))
        r.regressor_fit();self.assertTrue(all(0 not in m['indices'] for m in r.portable_models_))
        self.assertTrue(r.response_summary_['subset_counts'].get('nonfinite_response_features',0)>0)
    def test_compiled_finite_difference_constraints(self):
        variables=[dict(id=c,column=c,unit='1',domain=[None,None]) for c in ['x','z']]
        request=types.SimpleNamespace(X=self.X,y=self.y,variables=variables,target={'domain':[None,None]},limits={'response_checks':[dict(feature='x',step=.2,order=1,sign=-1,tolerance=0,points=[dict(x=0,z=0)])]})
        cp=compile_response_constraints(request)
        r=native('Regressor.py',self.X,self.y,['x','z'],ResponseSearch(cp,lambda names:np.asarray(cp['points'])))
        r.regressor_fit()
        for m in r.portable_models_:
            self.assertTrue(response_feasible(m['intercept']+np.asarray(cp['points'])[:,m['indices']]@m['coefficients'],cp))
    def test_actual_native_descriptor_replay(self):
        from ml_platform_symbolic.engines.torchsisso_oracle import descriptor_values
        variables=[dict(id=c,column=c,unit='1') for c in ['x','z']]
        cp=compiled([[-1,0],[1,0]],-.1,.1)
        names=['(x**2)','sin(z)']
        X=descriptor_values(names,self.X,variables).numpy()
        evaluate=lambda ns:descriptor_values(ns,np.asarray(cp['points']),variables).numpy()
        for file in ['Regressor.py','Regressor_dimension.py']:
            r=native(file,X,self.y,names,ResponseSearch(cp,evaluate))
            r.regressor_fit()
            for m in r.portable_models_:
                P=evaluate([r.names[i] for i in m['indices']])
                self.assertTrue(response_feasible(m['intercept']+P@m['coefficients'],cp))

    def test_source_coordinate_conversion_math(self):
        # Compiler consumes prepared working points, not raw/source points.
        # x_work=(x_source-10)/3; y_source=5*y_work+7.
        cp=compiled([[0,0]],(30-7)/5,(31-7)/5)
        r=native('Regressor.py',self.X,self.y,['x','z'],ResponseSearch(cp,lambda names:np.asarray(cp['points'])),dimension=1)
        r.regressor_fit();m=r.portable_models_[0]
        self.assertGreaterEqual(5*m['intercept']+7,30);self.assertLessEqual(5*m['intercept']+7,31)

if __name__=='__main__': unittest.main(verbosity=2)
