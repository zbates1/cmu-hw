#!/usr/bin/env python3
# minimal_pde.py — tiny PDE smoothing for PGM (P2) grayscale images
# Usage:
#   python minimal_pde.py foot.pgm linear  iters=25 dt=0.2
#   python minimal_pde.py foot.pgm pm      iters=25 dt=0.2 lam=10.0
import sys, numpy as np

def read_pgm_p2(path):
    with open(path,'r') as f:
        assert f.readline().strip()=='P2'
        line=f.readline().strip()
        while line.startswith('#') or len(line)==0:
            line=f.readline().strip()
        w,h = map(int,line.split())
        maxv = int(f.readline().strip())
        vals=[]
        for s in f.read().split():
            vals.append(int(s))
        img=np.array(vals,dtype=float).reshape(h,w)
        return img,maxv

def write_pgm_p2(path, img, maxv=255, comment="Created by minimal_pde.py"):
    img=np.clip(img,0,maxv)
    h,w=img.shape
    with open(path,'w') as f:
        f.write("P2\n# "+comment+"\n")
        f.write(f"{w} {h}\n{maxv}\n")
        flat=np.rint(img).astype(int).ravel()
        for i,v in enumerate(flat):
            f.write(str(v)+"\n" if (i+1)%17==0 else str(v)+" ")

def laplace(u):
    p=np.pad(u,1,mode='edge')
    return (p[2:,1:-1]+p[:-2,1:-1]+p[1:-1,2:]+p[1:-1,:-2]-4*u)

def heat(u,iters=25,dt=0.2):
    x=u.copy()
    for _ in range(iters):
        x+=dt*laplace(x)
    return x

def pm(u,iters=25,dt=0.2,lam=10.0):
    x=u.copy()
    for _ in range(iters):
        p=np.pad(x,1,mode='edge'); c=p[1:-1,1:-1]
        n=p[:-2,1:-1]-c; s=p[2:,1:-1]-c; w=p[1:-1,:-2]-c; e=p[1:-1,2:]-c
        g=lambda d: np.exp(-(np.abs(d)/lam)**2)
        x = x + dt*(g(n)*n + g(s)*s + g(w)*w + g(e)*e)
    return x

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python minimal_pde.py foot.pgm (linear|pm) [iters=25] [dt=0.2] [lam=10.0]")
        sys.exit(1)
    path,mode=sys.argv[1],sys.argv[2]
    kv={k:float(v) for (k,v) in (a.split('=') for a in sys.argv[3:] if '=' in a)}
    iters=int(kv.get('iters',25)); dt=kv.get('dt',0.2); lam=kv.get('lam',10.0)
    img,maxv=read_pgm_p2(path)
    out = heat(img,iters,dt) if mode=='linear' else pm(img,iters,dt,lam)
    stem=path.rsplit('.',1)[0]
    of=f"{stem}_{'linear' if mode=='linear' else f'pm_lam{lam:g}'}_it{iters}.pgm"
    write_pgm_p2(of,out,maxv)
    print("Wrote",of)
