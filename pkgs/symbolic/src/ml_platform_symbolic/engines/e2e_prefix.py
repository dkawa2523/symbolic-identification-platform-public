"""Pinned scalar-prefix grammar and source-expression completion budget.

No coefficient fitting, physical-unit validation, or response-condition guarantee.
"""
import math
import re
class PrefixMask:
    def __init__(self, env, max_nodes=None, *, variable_costs=None, output_cost=0, supported_operators=None):
        self.output_cost=output_cost
        self.variable_costs=variable_costs or {}
        self.tokens=env.equation_word2id;self.reverse={v:k for k,v in self.tokens.items()};self.max_nodes=max_nodes
        if env.float_encoder.mantissa_len != 1: raise ValueError('Pinned equation decoder only supports one mantissa token')
        self.operators=env.equation_encoder.all_operators
        self.actions={}
        for token,index in self.tokens.items():
            if token in self.operators and self.operators[token]>0 and (supported_operators is None or token in supported_operators):
                self.actions[index]=('expr',self.operators[token],2 if token in {'inv','pow2','pow3'} else 1)
            elif re.fullmatch(r'x_\d+',token):
                if token in self.variable_costs:self.actions[index]=('expr',0,self.variable_costs[token])
                elif variable_costs is None:self.actions[index]=('expr',0,5)
            elif token in {'+','-'}:self.actions[index]=('sign',0,1)
            elif re.fullmatch(r'N\d+',token):self.actions[index]=('mantissa',0,0)
            elif re.fullmatch(r'E-?\d+',token):self.actions[index]=('exponent',0,0)
            else:
                try: value=float(token)
                except ValueError:continue
                if math.isfinite(value):self.actions[index]=('expr',0,1)
        self.groups={}
        for index,action in self.actions.items():self.groups.setdefault(action,[]).append(index)
        self.rows=0
        self.masked_choices=0
    def step(self,state,index):
        pending,phase,cost=state;action=self.actions.get(index)
        if not action:return None
        kind,arity,weight=action
        if phase:
            if phase==2 and kind=='mantissa':return pending,1,cost
            if phase==1 and kind=='exponent':return pending,0,cost
            return None
        if pending==0:return None
        if kind=='sign':return pending-1,2,cost+1
        if kind=='expr':return pending-1+arity,0,cost+weight
        return None
    def __call__(self,prefixes,logits,remaining):
        if prefixes is None or remaining is None:raise ValueError('Prefix-aware decoder hook required')
        result=logits.clone();prefixes=prefixes.tolist();eos=self.tokens['<EOS>']
        for row,prefix in enumerate(prefixes):
            state=(1,0,self.output_cost);finished=False
            if not prefix or prefix[0]!=eos:raise ValueError('Expected native BOS/EOS prefix')
            for index in prefix[1:]:
                if finished:
                    if self.reverse[index]!='<PAD>':raise ValueError('Nonpadding token after EOS')
                    continue
                if index==eos:
                    if state[:2]!=(0,0):raise ValueError('Premature EOS in native expression')
                    finished=True;continue
                state=self.step(state,index)
                if state is None:raise ValueError('Invalid native scalar-prefix state')
            if finished or state[:2]==(0,0):
                if not math.isfinite(float(logits[row,eos])):raise ValueError('EOS unavailable')
                score=result[row,eos].clone();result[row,:]=-float('inf');result[row,eos]=score;continue
            allowed=[]
            # Use only leaves still allowed by operator/variable masks.
            finite=logits[row].isfinite().tolist()
            available={a:[i for i in indices if finite[i]] for a,indices in self.groups.items()}
            available={a:indices for a,indices in available.items() if indices}
            leaves=[a[2] for a in available if a[0]=='expr' and a[1]==0]
            kinds={a[0] for a in available}
            float_available={'sign','mantissa','exponent'}<=kinds
            for action,indices in available.items():
                next_state=self.step(state,indices[0])
                if next_state is None:continue
                pending,phase,cost=next_state
                if phase==2 and not {'mantissa','exponent'}<=kinds:continue
                if phase==1 and 'exponent' not in kinds:continue
                feasible=False
                for floats in range(pending+1):
                    singles=pending-floats
                    if floats and not float_available:continue
                    if singles and not leaves:continue
                    needed_tokens=phase+3*floats+singles+1
                    needed_nodes=cost+floats+(min(leaves)*singles if singles else 0)
                    if needed_tokens<=remaining-1 and (self.max_nodes is None or needed_nodes<=self.max_nodes):
                        feasible=True;break
                if not feasible:continue
                allowed.extend(indices)
            if not allowed:raise ValueError('No grammar completion fits remaining token/node budget')
            mask=result[row].new_full(result[row].shape,-float('inf'));mask[allowed]=result[row,allowed];result[row]=mask
        self.masked_choices += int((logits.isfinite() & ~result.isfinite()).sum().item())
        self.rows+=len(prefixes)
        return result
