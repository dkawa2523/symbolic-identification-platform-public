from copy import deepcopy
import pytest
from ml_platform_symbolic.engines.parfam_budget import lower_degree_families


def family(degree=2):
 return dict(degree_input_polynomials=0,degree_output_polynomials=degree,
             degree_input_denominator=0,degree_output_denominator=0,
             functions=[],function_names=[],width=1,maximal_potence=2,
             degree_output_polynomials_specific=None,enforce_function=False)


def test_missing_lower_degree_keeps_original_repeats_and_fixed_topology():
 original=[family(),family()]*2;before=deepcopy(original)
 result=lower_degree_families(original,'Complete',2)
 assert [x[0] for x in result[:4]]==original==before
 assert len(result)==6
 for p,origin in result[4:]:
  assert origin=='lower_degree_extension' and p['degree_output_polynomials']==1
  assert {k:v for k,v in p.items() if k!='degree_output_polynomials'}=={k:v for k,v in original[0].items() if k!='degree_output_polynomials'}
 assert lower_degree_families(original,'No',2)==[(p,'native') for p in original]


def test_existing_lower_family_not_duplicated_and_grid_stays_bounded():
 a=family();a.update(degree_input_polynomials=2,degree_input_denominator=1,degree_output_denominator=1,functions=['sin'],function_names=['sin'],degree_output_polynomials_specific=[2])
 result=lower_degree_families([a],'Max',1)
 assert len(result)==16
 assert all(p['functions']==['sin'] and p['degree_output_polynomials_specific']==[2] for p,_ in result)
 existing=[family(1),family(2)]*2
 assert lower_degree_families(existing,'Complete',2)==[(p,'native') for p in existing]


def test_invalid_repetition_layout_fails_closed():
 with pytest.raises(ValueError,match='repetition'):
  lower_degree_families([family(1),family(2)],'Complete',2)


def test_family_search_preserves_defaults_and_rejects_unknown():
 from ml_platform_symbolic.engines.parfam import family_search
 assert family_search({},'polynomial')=='fixed'
 assert family_search({},'rational')=='up_to'
 assert family_search({'family_search':'up_to'},'polynomial')=='up_to'
 assert family_search({'family_search':'fixed'},'rational')=='fixed'
 for invalid in (None, [], 'auto'):
  with pytest.raises(ValueError,match='family_search'):family_search({'family_search':invalid},'polynomial')
