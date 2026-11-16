"""Pinned constrained-SIS extension: finite hard response constraints *inside* SIS subset fitting.
No soft penalty, no unconstrained fallback. Native feature expansion/SIS retained.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, LinearConstraint, linprog
import torch
from collections import Counter
from ..response_search import response_feasible, inward_response_bounds

@dataclass
class ResponseSearch:
    compiled: dict
    evaluate_descriptors: object  # names -> values at compiled working-coordinate points
    max_iterations: int = 200

    def __post_init__(self):
        self.counts = Counter()
        self.failure_examples = []
        self.models = []

    def prepare(self, reg):
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not self.compiled["terms"]:
            raise ValueError("Response search requires at least one compiled condition")
        self.reg = reg
        self.features = np.asarray(self.evaluate_descriptors(reg.names), dtype=float)
        if self.features.shape != (len(self.compiled['points']), len(reg.names)):
            raise ValueError('Response feature shape mismatch after native column selection')
        self.counts = Counter()
        self.failure_examples = []
        self.models = []
        self.matrix = np.zeros((len(self.compiled['terms']), len(self.compiled['points'])))
        for row,t in zip(self.matrix,self.compiled['terms']):
            np.add.at(row,t['indices'],np.asarray(t['coefficients'])/t['scale'])
        self.lower=np.array([-np.inf if t['lower'] is None else t['lower']/t['scale'] for t in self.compiled['terms']])
        self.upper=np.array([np.inf if t['upper'] is None else t['upper']/t['scale'] for t in self.compiled['terms']])
        if np.any(self.lower>self.upper): raise ValueError('Contradictory response bounds')
        self.raw_x=reg.x.detach().cpu().numpy()
        self.z=reg.x_standardized.detach().cpu().numpy()
        self.y=reg.y.detach().cpu().numpy()
        self.xmean=reg.x_mean.detach().cpu().numpy()
        self.xstd=reg.x_std.detach().cpu().numpy()

    def feasible(self, pred):
        return response_feasible(pred, self.compiled)

    def solve(self, ids):
        result, event = self._solve(ids)
        self.counts[event['status']] += 1
        if result is not None:
            result['solver'] = event
        elif len(self.failure_examples) < 5:
            self.failure_examples.append(event)
        return result

    def _solve(self, ids):
        ids=np.asarray(ids,dtype=int)
        event={'indices':ids.tolist(),'status':'not_solved'}
        probe=self.features[:,ids]
        if not np.isfinite(probe).all():
            event['status']='nonfinite_response_features'; return None, event
        # beta[0] is an independently optimized standardized-feature intercept.
        X=np.column_stack([np.ones(len(self.y)),self.z[:,ids]])
        P=np.column_stack([np.ones(len(probe)),(probe-self.xmean[ids])/self.xstd[ids]])
        if not np.isfinite(X).all() or not np.isfinite(P).all():
            event['status']='nonfinite_design'; return None, event
        A=self.matrix@P
        if not np.isfinite(A).all():
            event['status']='nonfinite_constraint_design'; return None, event
        def unpack(b):
            c=b[1:]/self.xstd[ids]
            return float(b[0]-self.xmean[ids]@c),c
        def acceptable(b):
            intercept,c=unpack(b)
            train = intercept + self.raw_x[:, ids] @ c
            return bool(np.isfinite(train).all() and np.isfinite(np.mean((train-self.y)**2)) and (self.feasible_model(ids, intercept, c) if hasattr(self, "feasible_model") else self.feasible(intercept+probe@c)))
        scale=getattr(self, "objective_scale", max(float(np.std(self.y)),1e-12))
        def objective(b):
            r=(X@b-self.y)/scale
            return float(r@r/len(r))
        def jac(b): return 2*X.T@(X@b-self.y)/(len(self.y)*scale**2)
        initial=np.linalg.lstsq(X,self.y,rcond=None)[0]
        eq=self.lower==self.upper
        # Linear feasibility phase distinguishes an actually infeasible candidate
        # from solver failure; LP does not optimize prediction accuracy.
        U=[];v=[]
        for row,lo,hi,equal in zip(A,self.lower,self.upper,eq):
            if equal: continue
            if np.isfinite(hi): U.append(row);v.append(hi)
            if np.isfinite(lo): U.append(-row);v.append(-lo)
        lp=linprog(np.zeros(X.shape[1]),A_ub=np.asarray(U) if U else None,b_ub=v if U else None,
                   A_eq=A[eq] if eq.any() else None,b_eq=self.lower[eq] if eq.any() else None,
                   bounds=[(None,None)]*X.shape[1],method='highs',options={'maxiter':self.max_iterations})
        event['feasibility_solver_status']=int(lp.status)
        if not lp.success:
            event['status']='infeasible' if lp.status==2 else 'feasibility_solver_failed'
            event['infeasibility_is_solver_report_not_exact_proof']=True
            return None, event
        starts=[initial,lp.x]
        feasible=[]
        solver=[]
        optimized_feasible = False
        for start in starts:
            if acceptable(start): feasible.append((objective(start),start.copy(),False,'feasible_seed'))
            result=minimize(objective,start,jac=jac,method='SLSQP',
                constraints=[LinearConstraint(A,self.lower,self.upper)],
                options={'maxiter':self.max_iterations,'ftol':1e-12})
            accepted = acceptable(result.x)
            solver.append({'success':bool(result.success),'status':int(result.status),'iterations':int(result.nit),
                           'accepted_original_conditions':accepted})
            if accepted:
                feasible.append((objective(result.x),result.x.copy(),bool(result.success),'optimized'))
                optimized_feasible = True
        # If roundoff lands just outside an inequality, tighten only solver bounds.
        # Accept only original bounds. Equality-only boundary feasibility is not relaxed.
        # A feasible LP/initial seed is not an optimized solution. Retain it,
        # but do not let it suppress the one existing inward retry when every
        # optimizer return failed original-bound/full-AST acceptance.
        if not optimized_feasible:
            lo, hi = inward_response_bounds(self.lower, self.upper)
            result=minimize(objective,lp.x,jac=jac,method='SLSQP',constraints=[LinearConstraint(A,lo,hi)],options={'maxiter':self.max_iterations,'ftol':1e-12})
            accepted = acceptable(result.x)
            solver.append({'success':bool(result.success),'status':int(result.status),'iterations':int(result.nit),
                           'inward_retry':True,'accepted_original_conditions':accepted})
            if accepted:
                feasible.append((objective(result.x),result.x.copy(),bool(result.success),'inward_optimized'))
                optimized_feasible = True
        event.update(solver=solver, feasible_seed_count=sum(item[3]=='feasible_seed' for item in feasible),
                     optimized_candidate_accepted=optimized_feasible)
        if not feasible:
            event['status']='no_numerically_feasible_solution';return None, event
        _,beta,converged,origin=min(feasible,key=lambda item:item[0])
        intercept,c=unpack(beta)
        residual=self.y-(intercept+self.raw_x[:,ids]@c)
        mse=float(np.mean(residual**2))
        event.update(status='feasible',mse=mse,optimizer_converged=converged,
                     solution_origin=origin,optimized_candidate_accepted=optimized_feasible)
        return dict(indices=ids.tolist(),coefficients=c.tolist(),intercept=intercept,
                    term_count=len(ids),mse=mse,optimizer_converged=converged,residual=residual,
                    solution_origin=origin), event

    def choose(self, combinations):
        best = None
        for ids in combinations:
            result = self.solve(ids)
            if result is not None and (best is None or result['mse'] < best['mse']):
                best = result
        if best is None: return None
        self.reg.residual=torch.as_tensor(best['residual'],dtype=self.reg.y.dtype,device=self.reg.device).reshape(1,-1)
        self.models.append({k:v for k,v in best.items() if k!='residual'})
        return best


def higher(reg,combinations):
    best=reg.response_search.choose(combinations.detach().cpu().tolist())
    if best is None:return None
    c=torch.tensor(best['coefficients'],dtype=reg.y.dtype,device=reg.device).reshape(1,-1)
    terms=[f'{coef:.17g}*{reg.names[i]}' for i,coef in zip(best['indices'],best['coefficients'])]
    total=float(torch.sum(reg.y_centered**2))
    r2=1-best['mse']*len(reg.y)/total if total>0 else float('nan')
    return best['mse']**.5,terms,best['intercept'],c,r2


def fit(reg):
    response=reg.response_search
    response.prepare(reg)
    # Explicit constrained-SIS extension: fit ALL 1-term SIS candidates, not just
    # unconstrained correlation winner. Native multivariate SIS screening follows.
    reg.sis_features=min(reg.sis_features,reg.x.shape[1])
    count=reg.dimension*reg.sis_features
    reg.indices=torch.arange(1,count+1,device=reg.device).reshape(count,1)
    scores=abs(reg.y_centered.reshape(1,-1)@reg.x_standardized)
    scores[torch.isnan(scores)]=0
    ids=torch.topk(scores,k=reg.sis_features).indices.T
    padded=torch.cat([ids,torch.full((count-len(ids),1),float('nan'),device=reg.device)])
    reg.indices=torch.cat([reg.indices,padded],dim=1)
    # If first-order feasible set is empty, keep centered-y screening residual;
    # do not stop: two-term structures can be feasible when no one-term is.
    reg.residual=reg.y_centered.reshape(1,-1).clone()
    response.choose(ids.detach().cpu().tolist())
    for iteration in range(2,reg.dimension+1):
        reg.higher_dimension(iteration)  # patched only at native LS selection site
    reg.portable_models_=response.models
    reg.response_summary_={"subset_counts": dict(response.counts), "failure_examples": response.failure_examples}
    if not response.models:raise ValueError('No feasible model in explored SIS combinations')
    equations=[' + '.join([f'{c:.17g}*{reg.names[i]}' for i,c in zip(m['indices'],m['coefficients'])]+[f"{m['intercept']:.17g}"]) for m in response.models]
    last=response.models[-1]
    total=float(torch.sum(reg.y_centered**2))
    r2=1-last['mse']*len(reg.y)/total if total>0 else float('nan')
    return last['mse']**.5,equations[-1],r2,equations


def instrument(source):
    """Add two hooks after the pinned budget patch, preserving the default path."""
    replacements = [
        ('node_budget=None):', 'node_budget=None,response_search=None):'),
        ('        self.node_budget = node_budget', '        self.response_search = response_search\n        self.node_budget = node_budget'),
        ('    def regressor_fit(self):', '    def regressor_fit(self):\n        if self.response_search is not None:\n            from ml_platform_symbolic.engines.torchsisso_response import fit\n            return fit(self)')]
    for before, after in replacements:
        if source.count(before) != 1: raise ValueError('Pinned constrained-SIS patch context differs')
        source = source.replace(before, after)
    marker = '        y_centered_clone = '
    if source.count(marker) != 1: raise ValueError('Pinned constrained-SIS solve boundary differs')
    pos = source.index(marker)
    source = source[:pos] + '        if self.response_search is not None:\n            from ml_platform_symbolic.engines.torchsisso_response import higher\n            return higher(self, combinations_generated)\n\n' + source[pos:]
    return source
