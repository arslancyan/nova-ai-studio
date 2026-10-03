"""Evaluate reconstruction or generated video tensors.

Reports pixel MSE/L1/PSNR and a simple temporal-difference consistency metric.
Metrics are descriptive diagnostics, not a measure of human visual quality.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F

def metrics(reference,prediction):
    reference=reference.float(); prediction=prediction.float().clamp(-1,1)
    mse=float(F.mse_loss(prediction,reference)); l1=float(F.l1_loss(prediction,reference))
    psnr=float(-10.0*torch.log10(torch.tensor(max(mse,1e-12))))
    ref_delta=reference[:,:,1:]-reference[:,:,:-1]; pred_delta=prediction[:,:,1:]-prediction[:,:,:-1]
    temporal_l1=float(F.l1_loss(pred_delta,ref_delta)) if reference.shape[2]>1 else 0.0
    return {"mse":mse,"l1":l1,"psnr_db":psnr,"temporal_delta_l1":temporal_l1}

def evaluate(reference_path,prediction_path,out=None):
    ref=torch.load(reference_path,map_location="cpu",weights_only=True)
    pred=torch.load(prediction_path,map_location="cpu",weights_only=True)
    if ref.ndim==4: ref=ref.unsqueeze(0)
    if pred.ndim==4: pred=pred.unsqueeze(0)
    if ref.shape!=pred.shape:
        pred=F.interpolate(pred,size=ref.shape[2:],mode="trilinear",align_corners=False)
    result=metrics(ref,pred); payload={"reference":str(reference_path),"prediction":str(prediction_path),"metrics":result}
    if out: Path(out).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2)); return payload

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--reference",required=True); p.add_argument("--prediction",required=True); p.add_argument("--out",default=None)
    a=p.parse_args(); evaluate(a.reference,a.prediction,a.out)
