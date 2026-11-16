from pathlib import Path
import pytest
from ml_platform_symbolic.engines import pysr_recovery as r
from ml_platform_symbolic.data import write_json

GOOD=b'Complexity,Loss,Equation\n1,1.0,x0\n3,0.0,x0 + 1.0\n'

def setup(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'version',lambda name:'2.7.2')
 train=tmp_path/'train.npz';train.write_bytes(b'fixed train')
 write_json(tmp_path/'job.json',{'train':str(train)})
 run=r.begin_run(tmp_path); d=tmp_path/'native'/run;d.mkdir(parents=True)
 for name in ['hall_of_fame.csv','hall_of_fame.csv.bak']:(d/name).write_bytes(GOOD)
 return d

def test_recovery_fresh_identity_and_immutable_csv(tmp_path,monkeypatch):
 d=setup(tmp_path,monkeypatch);(d/'checkpoint.pkl').write_bytes(b'never unpickle')
 snapshot,evidence=r.recovered_run(tmp_path)
 assert evidence['rows']==2 and (snapshot/'hall_of_fame.csv').read_bytes()==GOOD
 assert not (snapshot/'checkpoint.pkl').exists()
 assert (d/'checkpoint.pkl').read_bytes()==b'never unpickle'
 with pytest.raises(FileExistsError):r.recovered_run(tmp_path)

@pytest.mark.parametrize('change',['train','job','run_id','generation','truncated','missing'])
def test_recovery_rejects_stale_corrupt_ambiguous(tmp_path,monkeypatch,change):
 d=setup(tmp_path,monkeypatch)
 if change=='train':(tmp_path/'train.npz').write_bytes(b'other run')
 elif change=='job':write_json(tmp_path/'job.json',{'train':str(tmp_path/'train.npz'),'seed':99})
 elif change=='run_id':
  m=r.read_json(tmp_path/'pysr_run.json');m['run_id']='../other';write_json(tmp_path/'pysr_run.json',m)
 elif change=='generation':(d/'hall_of_fame.csv.bak').write_bytes(GOOD.replace(b'1.0,x0',b'2.0,x0'))
 elif change=='truncated':(d/'hall_of_fame.csv').write_bytes(GOOD[:-1])
 else:(d/'hall_of_fame.csv').unlink()
 with pytest.raises(ValueError):r.recovered_run(tmp_path)
 assert not (tmp_path/'recovery_csv').exists()

@pytest.mark.parametrize('content',[b'',GOOD[:-1],GOOD+b'3,0,x0\n',GOOD.replace(b'1.0,x0',b'nan,x0'),b'Complexity,Loss,Equation\n1,0,\n'])
def test_csv_rejects_invalid(content):
 with pytest.raises(ValueError):r.validate_csv(content)

def test_csv_infinite_objective_is_not_feasible():
 assert r.validate_csv(GOOD.replace(b'1.0,x0',b'Inf,x0'))==2


@pytest.mark.parametrize('header', [
    'Complexity,Loss,Equation,Equation',
    'Complexity,Loss,Loss,Equation',
    'Loss,Complexity,Equation',
    'Complexity,Loss,Equation,Other',
])
def test_csv_rejects_ambiguous_headers_before_dataframe_parser(header):
    payload=(header+'\n1,0,x0,x1\n').encode()
    with pytest.raises(ValueError, match='three unique ordered columns'):
        r.validate_csv(payload)
