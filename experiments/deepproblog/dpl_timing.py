"""Feasibility test: the FIGO program in DeepProbLog (ExactEngine) on real CTU-UHB inputs."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, torch, torch.nn as nn
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from deepproblog.dataset import DataLoader, Dataset
from deepproblog.optimizer import SGD
from deepproblog.train import train_model
from problog.logic import Term, Constant, Var
from src.data.inputs import signal_inputs, labels
from src.models.nets import CNNEncoder
from src.models.nesy import BASELINE, VARIAB, BINARY

recs, X = signal_inputs("W60", 1.0)
y = labels("y_ph705").reindex(recs).values.astype(int)
N = 64
Xt = torch.tensor(X[:N])

class Head(nn.Module):
    def __init__(self, k):
        super().__init__(); self.enc = CNNEncoder(4, 16, 3, 7, 0.2); self.fc = nn.Linear(self.enc.out_dim, k); self.k = k
    def forward(self, x):
        z = self.fc(self.enc(x)[0])
        return torch.softmax(z, -1) if self.k > 1 else torch.sigmoid(z)

lines = [f"nn(net_baseline,[X],Y,[{','.join(BASELINE)}]) :: baseline(X,Y).",
         f"nn(net_variab,[X],Y,[{','.join(VARIAB)}]) :: variability(X,Y)."]
nets = [Network(Head(4), "net_baseline", batching=True), Network(Head(4), "net_variab", batching=True)]
for b in BINARY:
    lines.append(f"nn(net_{b},[X]) :: {b}(X)."); nets.append(Network(Head(1), f"net_{b}", batching=True))
lines += ["pathological(X) :- baseline(X,sevbrady).", "pathological(X) :- reduced_var_long(X).",
          "pathological(X) :- increased_var_long(X).", "pathological(X) :- variability(X,sinusoidal).",
          "pathological(X) :- rep_late_long(X).", "pathological(X) :- very_prolonged(X).",
          "normal(X) :- baseline(X,normal), variability(X,normal), \+rep_any(X), \+prolonged(X), \+pathological(X).",
          "suspicious(X) :- \+pathological(X), \+normal(X).",
          "t(0.05) :: acid(X).", "t(0.1) :: acid(X) :- pathological(X).", "t(0.1) :: acid(X) :- suspicious(X).",
          "t(0.1) :: acid(X) :- rep_late(X).", "t(0.1) :: acid(X) :- rep_variable(X).", "t(0.1) :: acid(X) :- prolonged(X).",
          "t(0.1) :: acid(X) :- baseline(X,tachy).", "t(0.1) :: acid(X) :- \+accel(X).",
          "t(0.1) :: acid(X) :- variability(X,reduced).", "t(0.1) :: acid(X) :- rep_any(X)."]
prog = ROOT / "experiments/deepproblog/figo_dpl.pl"; prog.write_text("\n".join(lines))
for n in nets: n.optimizer = torch.optim.Adam(n.parameters(), lr=1e-3)
model = Model(str(prog), nets); model.set_engine(ExactEngine(model))
class TS:
    def __getitem__(self, i): return Xt[int(i[0].value)]
model.add_tensor_source("train", TS())
class DS(Dataset):
    def __len__(self): return N
    def to_query(self, i):
        sub = {Term("a"): Term("tensor", Term("train", Constant(i)))}
        q = Term("acid", Term("a"))
        return Query(q, sub, p=float(y[i]))
model.optimizer = SGD(model, 1e-3)
loader = DataLoader(DS(), 8, False)
t = time.time()
try:
    train_model(model, loader, 1, log_iter=1000, profile=0)
    dt = time.time() - t
    print(f"DeepProbLog: 1 epoch over {N} records took {dt:.1f} s ({dt/N*1000:.0f} ms/record)")
except Exception as e:
    import traceback; traceback.print_exc(); print("DeepProbLog run FAILED after", round(time.time()-t,1), "s")
