"""Pinned TorchSISSO extension: restrict affine candidate fits by source AST size."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

@dataclass(frozen=True)
class NodeBudget:
    max_nodes: int
    descriptor_cost: object
    def __post_init__(self):
        if type(self.max_nodes) is not int or self.max_nodes < 1:
            raise ValueError('max_nodes must be a positive integer')
    def costs(self,names):
        costs=[self.descriptor_cost(name) for name in names]
        if any(type(x) is not int or x<1 for x in costs):
            raise ValueError('descriptor costs must be positive integers')
        return costs
    def admissible(self,indices,costs):
        return 1+sum(costs[int(i)]+3 for i in indices)<=self.max_nodes

def prepare(regressor):
    budget=regressor.node_budget
    if budget is None:return
    costs=budget.costs(regressor.names)
    keep=[i for i in range(len(costs)) if budget.admissible([i],costs)]
    if not keep:raise ValueError('No descriptor fits affine output node budget before fitting')
    regressor.x=regressor.x[:,keep]
    regressor.names=[regressor.names[i] for i in keep]
    regressor.node_costs=[costs[i] for i in keep]
    if hasattr(budget, "select"):
        budget.select(keep)
    regressor.dimension=min(regressor.dimension,(budget.max_nodes-1)//(min(regressor.node_costs)+3))
    regressor.node_budget_events=[]

def feasible_combinations(regressor,combinations):
    if regressor.node_budget is None:return combinations
    import torch
    allowed=[regressor.node_budget.admissible(row,regressor.node_costs) for row in combinations.detach().cpu().tolist()]
    regressor.node_budget_events.append({'proposed':len(allowed),'feasible':sum(allowed)})
    return combinations[torch.tensor(allowed,dtype=torch.bool,device=combinations.device)]

HASHES={'Regressor.py':'a14560c8cea5d001914bf13986ba6f629a68ccb36f974db2f3d89d49992f7a63','Regressor_dimension.py':'cc0ea1b83c11f250a74f9910e49f94e8ac86f586ab04a994500cacc5b73df07d'}
def patch_source(data,filename):
    if hashlib.sha256(data).hexdigest()!=HASHES[filename]:raise ValueError('Unreviewed TorchSISSO native source')
    s=data.decode()
    replacements=[
      ('class Regressor:', 'class Regressor:\n    symbolic_handoff_budget_version = 1'),
      ('use_gpu=False):','use_gpu=False,node_budget=None):'),
      ('        self.device = device','        self.node_budget = node_budget\n        self.device = device'),
      ('        # Transform the features into standardized format','        from ml_platform_symbolic.engines.torchsisso_budget import prepare\n        prepare(self)\n\n        # Transform the features into standardized format'),
      ('        combinations_generated = torch.combinations(comb1,(int(self.indices.shape[1])-1))','        combinations_generated = torch.combinations(comb1,(int(self.indices.shape[1])-1))\n        from ml_platform_symbolic.engines.torchsisso_budget import feasible_combinations\n        combinations_generated = feasible_combinations(self, combinations_generated)\n        if len(combinations_generated) == 0:\n            return None'),
      ('                rmse,terms,intercept,coefs,r2 = self.higher_dimension(i)','                result = self.higher_dimension(i)\n                if result is None:\n                    break\n                rmse,terms,intercept,coefs,r2 = result')]
    for before,after in replacements:
        if s.count(before)!=1:raise ValueError('Pinned TorchSISSO budget patch context differs')
        s=s.replace(before,after)
    from .torchsisso_response import instrument as response_instrument
    s = response_instrument(s)
    if filename=='Regressor.py':
        from ml_platform_symbolic.engines.torchsisso_precision import instrument
        s=instrument(s)
    compile(s,filename,'exec')
    return s.encode()

def patch_package(directory):
    from ml_platform_symbolic.engines.torchsisso_precision import instrument
    plans=[]
    for name in HASHES:
        path=Path(directory)/name;current=path.read_bytes();original=current
        if hashlib.sha256(current).hexdigest()!=HASHES[name]:
            saved=path.with_suffix('.py.original')
            if not saved.exists():raise ValueError('Unrecognized native source; original source unavailable')
            original=saved.read_bytes()
            expected=instrument(original.decode()).encode() if name=='Regressor.py' else original
            updated=patch_source(original,name)
            approved_previous = {'Regressor.py': '71495b48c814ed4611cc8d43e66b4b5ff772f4d487e0790dc5cc27f66e6268b2', 'Regressor_dimension.py': '0316396538d4ed6aedf3790c71d81b9dda60f50974199598bfc858be56dd879f'}
            if current not in (expected,updated) and hashlib.sha256(current).hexdigest() != approved_previous[name]:raise ValueError('Native source differs from approved instrumentation')
        updated=patch_source(original,name)
        plans.append((path,original,updated))
    for path,original,updated in plans:
        saved=path.with_suffix(".py.original")
        if not saved.exists():saved.write_bytes(original)
        path.write_bytes(updated)
        print(path.name,hashlib.sha256(updated).hexdigest())

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--package-dir',required=True);args=p.parse_args();patch_package(args.package_dir)
