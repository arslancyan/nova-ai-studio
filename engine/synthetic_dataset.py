"""Create a fully synthetic, rights-safe NOVA training dataset.

The clips contain simple moving geometric objects. Captions describe the
generated scene. No external media or pretrained model is used.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

COLORS={"red":(0.9,0.15,0.15),"green":(0.2,0.9,0.2),"blue":(0.15,0.35,0.95),"yellow":(0.95,0.8,0.1)}
SHAPES=("circle","square")

def make_clip(frames=8,height=32,width=32,seed=0):
    g=torch.Generator().manual_seed(seed)
    color_name=list(COLORS)[int(torch.randint(0,len(COLORS),(1,),generator=g))]
    shape=SHAPES[int(torch.randint(0,len(SHAPES),(1,),generator=g))]
    color=torch.tensor(COLORS[color_name],dtype=torch.float32).view(3,1,1)
    x0=int(torch.randint(5,width-10,(1,),generator=g)); y0=int(torch.randint(5,height-10,(1,),generator=g))
    vx=int(torch.randint(-2,3,(1,),generator=g)); vy=int(torch.randint(-2,3,(1,),generator=g))
    size=int(torch.randint(4,8,(1,),generator=g))
    clip=torch.zeros(3,frames,height,width)
    yy,xx=torch.meshgrid(torch.arange(height),torch.arange(width),indexing="ij")
    for t in range(frames):
        cx=max(size,min(width-size-1,x0+vx*t)); cy=max(size,min(height-size-1,y0+vy*t))
        if shape=="circle": mask=(xx-cx)**2+(yy-cy)**2<=size**2
        else: mask=((xx-cx).abs()<=size) & ((yy-cy).abs()<=size)
        clip[:,t][mask]=color[:,0,0].view(3,1)
    return clip*2-1, f"a {color_name} {shape} moving in a dark scene"

def build_dataset(samples=256,frames=8,height=32,width=32,output="data/synthetic"):
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    clips=[]; captions=[]
    for i in range(samples):
        clip,caption=make_clip(frames,height,width,i)
        clips.append(clip); captions.append(caption)
    torch.save(torch.stack(clips),root/"clips.pt")
    (root/"captions.txt").write_text("\n".join(captions)+"\n",encoding="utf-8")
    print(f"saved {len(clips)} clips to {root}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--samples",type=int,default=256); p.add_argument("--output",default="data/synthetic")
    a=p.parse_args(); build_dataset(a.samples,output=a.output)
