"""Synthetic banking and telecom cases for reproducible regression diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def banking_credit_case(n=1200, seed=42):
    rng=np.random.default_rng(seed)
    income=rng.lognormal(8.8,.45,n)
    utilization=np.clip(rng.beta(2.2,4.5,n),.01,.98)
    debt=income*(.4+1.4*utilization)+rng.normal(0,1800,n)
    amount=6000+2.1*income+0.7*debt+rng.normal(0,9000,n)
    return pd.DataFrame({"income":income,"utilization":utilization,"debt":debt,"requested_amount":amount})


def telecom_quality_case(n=900, seed=42):
    rng=np.random.default_rng(seed)
    latency=np.clip(rng.normal(55,15,n),5,None)
    jitter=np.clip(0.35*latency+rng.normal(8,6,n),0,None)
    loss=np.clip(rng.beta(1.5,25,n)*100,0,15)
    nps=75-.55*latency-.8*jitter-2.5*loss+rng.normal(0,8,n)
    return pd.DataFrame({"latency_ms":latency,"jitter_ms":jitter,"packet_loss_pct":loss,"nps":nps})
