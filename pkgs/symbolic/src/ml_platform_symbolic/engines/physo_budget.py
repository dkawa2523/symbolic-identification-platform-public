"""Strict source-size prior appended to pinned PhySO collections before sampling."""
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np

@contextmanager
def source_budget_prior(max_nodes, variable_costs, output_cost, supported_operators, audit, token_multiplier=1):
    import physo.physym.prior as native
    prior_sha256 = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if prior_sha256 not in {'adf0a33f9bb6e9625120ac6700a5c4ef2cbd20df492dbedf152c6f3e4f7ef734', '024fb6df7341929dc27be33a3ba222d034c47a733716ade6edcea50aa047c885'}:
        raise ValueError('Unsupported PhySO prior source; requires reviewed pinned 1.1.11')
    if type(max_nodes) is not int or max_nodes < output_cost+1:
        raise ValueError('PhySO source budget cannot fit output coordinate map')
    from importlib import import_module
    learning = import_module("physo.learn.learn")
    if hashlib.sha256(Path(learning.__file__).read_bytes()).hexdigest() != "65bfac41ea384865c87388c157d7adc4fb86635d22b1d5967304e8f37054a4a8":
        raise ValueError("Unreviewed PhySO sampler; exact-zero hard prior behavior is not verified")
    original=native.make_PriorCollection
    class SourceBudget(native.Prior):
        def __init__(self,library,programs):
            super().__init__(library,programs)
            weights=[]
            for token in library.lib_tokens[:library.n_choices]:
                if token.var_type==1:
                    weights.append(variable_costs[int(token.var_id)])
                elif token.var_type in {3,20,21}:
                    weights.append(token_multiplier)
                elif token.var_type==0 and token.name in supported_operators:
                    weights.append(token_multiplier * (2 if token.name in {'n2','n3','n4','cbrt'} else 1))
                else:raise ValueError('Unsupported PhySO source-budget token: '+token.name)
            self.weights=np.asarray(weights,dtype=int)
            self.arities=np.asarray([t.arity for t in library.lib_tokens[:library.n_choices]],dtype=int)
            terminals=self.weights[self.arities==0]
            if not len(terminals):raise ValueError('PhySO grammar has no finite completion terminal')
            self.minimum_terminal=int(terminals.min())
            if max_nodes < output_cost+self.minimum_terminal:
                raise ValueError("PhySO source budget cannot fit any terminal")
            self.record={'prior_source_sha256':prior_sha256, 'token_costs':{t.name:int(w) for t,w in zip(library.lib_tokens[:library.n_choices],weights)},'calls':0,'masked_choices':0}
            audit.append(self.record)
        def __call__(self):
            result=self.get_default_mask_prob()
            # Native append checks capacity BEFORE replacing completed-row actions
            # with invalid padding. Restrict only ignored padding actions here.
            for row in np.flatnonzero(self.progs.is_complete):
                room = self.progs.max_time_step - self.progs.n_lengths[row] - self.progs.n_dummies[row]
                result[row] = (self.arities <= room).astype(float)
            for row in np.flatnonzero(~self.progs.is_complete):
                prefix=self.progs.tokens.idx[row,:self.progs.n_lengths[row]]
                if np.any(prefix<0) or np.any(prefix>=len(self.weights)):
                    raise ValueError('Incomplete PhySO prefix includes nonchoice token')
                spent=int(self.weights[prefix].sum())
                remaining=int(self.progs.n_dangling[row])-1+self.arities
                required=output_cost+spent+self.weights+remaining*self.minimum_terminal
                result[row]=(required<=max_nodes).astype(float)
            self.record['calls']+=1
            self.record['masked_choices']+=int((result==0).sum())
            return result
    def build(library,programs,priors_config):
        collection=original(library,programs,priors_config)
        wanted=Counter(name for name,_ in priors_config)
        present=Counter(collection.priors_names)
        if wanted-present:
            raise ValueError('PhySO omitted requested prior(s): '+','.join((wanted-present).keys()))
        # Construct outside native's warning-and-ignore exception handler.
        prior=SourceBudget(library,programs)
        collection.set_priors([prior],['SourceNodeBudget'])
        if not any(p is prior for p in collection.priors):
            raise RuntimeError('PhySO source budget prior was not installed')
        return collection
    native.make_PriorCollection=build
    try:
        yield
    finally:
        native.make_PriorCollection=original
