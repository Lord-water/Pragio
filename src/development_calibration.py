"""Shared calibration functions from the frozen research implementation."""
import numpy as np
from evaluate_acceptability import LABELS
KINDS=("raw","temperature","affine_sigmoid")
SLOPE_BOUNDS=(0.0001,100.)

def sigmoid(z): return np.exp(-np.logaddexp(0.,-z))

def logit(p):
    p=np.clip(p,1e-7,1-1e-7)
    return np.log(p)-np.log1p(-p)

def fit(p,y,kind):
    if kind=='raw': return {'slope':1.,'intercept':0.,'kind':kind}
    assert kind in KINDS
    z=logit(p); known=y>=0
    counts=known.sum(0); eligible=counts>0
    w=np.divide(known,counts[None,:],out=np.zeros_like(z),where=counts[None,:]>0)/eligible.sum()
    zz=z[known]; yy=y[known]; ww=w[known]
    assert ((yy>=0)&(yy<=1)).all() and (yy>0).any() and (yy<1).any()

    def best_intercept(a):
        if kind=='temperature': return 0.
        low,high=-2000.,2000.
        for _ in range(70):
            middle=(low+high)/2
            gradient=np.sum(ww*(sigmoid(a*zz+middle)-yy))
            if gradient>0: high=middle
            else: low=middle
        return (low+high)/2

    def slope_gradient(a):
        b=best_intercept(a)
        return float(np.sum(ww*(sigmoid(a*zz+b)-yy)*zz)),b

    low,high=SLOPE_BOUNDS
    gl,_=slope_gradient(low); gh,_=slope_gradient(high)
    if gl>=0: a=low
    elif gh<=0: a=high
    else:
        for _ in range(65):
            a=(low+high)/2
            gradient,_=slope_gradient(a)
            if gradient>0: high=a
            else: low=a
        a=(low+high)/2
    ga,b=slope_gradient(a)
    gb=float(np.sum(ww*(sigmoid(a*zz+b)-yy)))
    loss=float(np.sum(ww*(np.logaddexp(0.,a*zz+b)-yy*(a*zz+b))))
    boundary='lower' if a==SLOPE_BOUNDS[0] else 'upper' if a==SLOPE_BOUNDS[1] else None
    kkt=abs(ga) if boundary is None else max(0.,-ga) if boundary=='lower' else max(0.,ga)
    if kind=='affine_sigmoid': kkt=max(kkt,abs(gb))
    if kkt>1e-7: raise ValueError(f'Optimizer failed KKT check: {kkt}')
    return {'kind':kind,'slope':float(a),'intercept':float(b),'temperature':float(1/a),
            'fit_macro_BCE':loss,'known_training_judgments':int(known.sum()),
            'slope_boundary':boundary,'gradient_slope':ga,'gradient_intercept':gb,'kkt_residual':kkt}

def transform(p,params):
    if params['kind']=='raw': return np.array(p,copy=True)
    return sigmoid(params['slope']*logit(p)+params['intercept'])

def loss_metrics(p,y):
    known=y>=0; n=known.sum(0)
    squared=((p-y)**2)*known
    bce=(-y*np.log(np.maximum(p,1e-7))-(1-y)*np.log(np.maximum(1-p,1e-7)))*known
    def mean(values):
        return np.divide(values.sum(0),n,out=np.full(p.shape[1],np.nan),where=n>0)
    bs=mean(squared); ce=mean(bce)
    return {'macro_brier':float(np.nanmean(bs)),'macro_bce':float(np.nanmean(ce)),
            'known':int(known.sum()),'per_label':{label:{'known':int(n[j]),'brier':float(bs[j]) if n[j] else None,
             'bce':float(ce[j]) if n[j] else None} for j,label in enumerate(LABELS)}}
