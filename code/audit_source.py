"""
audit_source.py -- identity-level audit of the mixed-order source pieces.

Needs no data files. Put it next to the project modules and run
    python audit_source.py
Each check compares the first-order-in-h formula used in the pipeline with a
central finite difference of the exact nonlinear expression evaluated on
g0 + eps*h, for a synthetic smooth even h and synthetic scalars on Kerr.
The first-order expansion of a tensor in a metric perturbation is an identity
valid for any h, so agreement does not depend on h solving anything.

Result on the project code as of 2026-10-05 (spin 0.6, 20x20):
    G1_lin, box_pert, pontryagin_pert, G2_bilinear   agree with the exact result
    gb_invariant_pert   misses  -1/2 h GB^(0)
    H_mixed             misses  the index-raising terms and the Ricci terms
    Eq. (A6)            has the wrong sign on its six Ricci-dependent terms (checked against
                        -4 (**R) nabla nabla phi built directly from the double dual)
    C_mixed             misses  the index-raising terms (and the grad-Ricci term,
                        which is not tested here)
    stress_cross        misses  the metric-perturbation part of T
Revised 2026-10-05: the first version of this script took the Ricci-term signs from Eq. (A6)
for both the reference and the fix, so it could not see that they were wrong.
"rel" printed below is max|code - exact| / max|exact| over bulk nodes.
"""
import numpy as np
from spectral_grid import SpectralGrid
from kerr_background import KerrBackground, _LC
from perturbation import (assemble_h, gb_invariant_pert, pontryagin_pert, box_pert,
                          covariant_derivs_h)
from correction_tensors import covariant_hessian, H_sGB_leading, C_dCS_leading
from stage2_source import H_mixed, C_mixed, stress_cross, _bg_d2g, G2_bilinear
from einstein_operator import G1_lin, stress_energy_scalar

def geom(bg, gd, dg, d2g):
    good = bg.interior
    gi = np.full_like(gd, np.nan); gi[good] = np.linalg.inv(gd[good])
    dgi = -np.einsum("...ma,...cab,...bn->...cmn", gi, dg, gi)
    Gam = 0.5*(np.einsum("...ls,...mns->...lmn", gi, dg)+np.einsum("...ls,...nms->...lmn", gi, dg)
               -np.einsum("...ls,...smn->...lmn", gi, dg))
    dGam = 0.5*(np.einsum("...cls,...mns->...clmn", dgi, dg)+np.einsum("...cls,...nms->...clmn", dgi, dg)
                -np.einsum("...cls,...smn->...clmn", dgi, dg)+np.einsum("...ls,...cmns->...clmn", gi, d2g)
                +np.einsum("...ls,...cnms->...clmn", gi, d2g)-np.einsum("...ls,...csmn->...clmn", gi, d2g))
    Ru = (np.einsum("...mrns->...rsmn", dGam)-np.einsum("...nrms->...rsmn", dGam)
          +np.einsum("...rml,...lns->...rsmn", Gam, Gam)-np.einsum("...rnl,...lms->...rsmn", Gam, Gam))
    Rd = np.einsum("...ar,...rbcd->...abcd", gd, Ru)
    Ric = np.einsum("...msmn->...sn", Ru)
    Rs = np.einsum("...ab,...ab->...", gi, Ric)
    det = np.linalg.det(np.where(np.isnan(gd), 0.0, gd))
    with np.errstate(all="ignore"):
        eu = (1.0/np.sqrt(np.abs(det)))[..., None, None, None, None]*_LC
    euudd = np.einsum("...abef,...ce,...df->...abcd", eu, gd, gd)
    Ruudd = np.einsum("...ai,...bj,...ijcd->...abcd", gi, gi, Rd)
    dual_uudd = 0.5*np.einsum("...abef,...efcd->...abcd", euudd, Ruudd)
    dual_dddd = np.einsum("...ai,...bj,...ijcd->...abcd", gd, gd, dual_uudd)
    return dict(gi=gi, Gam=Gam, Rd=Rd, Ric=Ric, Rs=Rs, dual_uudd=dual_uudd, dual_dddd=dual_dddd, Ruudd=Ruudd, euudd=euudd)

def hess_lower(bg, G, c):
    g = bg.grid
    grad = np.zeros(bg.shape+(4,)); grad[...,1]=g.d_dr(c); grad[...,2]=g.d_dchi(c)
    d2 = np.zeros(bg.shape+(4,4)); d2[...,1,1]=g.d2_dr2(c); d2[...,2,2]=g.d2_dchi2(c)
    d2[...,1,2]=d2[...,2,1]=g._dz_dr[:,None]*g.d_dz_dchi(c)
    return d2-np.einsum("...lmn,...l->...mn", G["Gam"], grad), grad

def GB(G):
    gi=G["gi"]; Rd=G["Rd"]
    Ruuuu=np.einsum("...ai,...bj,...ck,...dl,...ijkl->...abcd",gi,gi,gi,gi,Rd)
    Ricuu=np.einsum("...ma,...nb,...ab->...mn",gi,gi,G["Ric"])
    return np.einsum("...abcd,...abcd->...",Rd,Ruuuu)-4*np.einsum("...ab,...ab->...",G["Ric"],Ricuu)+G["Rs"]**2

def PONT(G):
    Ruucd_dd=np.einsum("...abcd->...abcd",G["Ruudd"])           # R^{ab}_{cd}
    # *RR = *R^{ab}{}_{cd} R^{cd}{}_{ab}
    return np.einsum("...abcd,...cdab->...",G["dual_uudd"],G["Ruudd"])

def BOX(bg,G,c):
    H,_=hess_lower(bg,G,c); return np.einsum("...mn,...mn->...",G["gi"],H)

def Hterms(bg,G,gd,c,ricci_sign=+1.0):
    """Seven-term form of H_{mu nu}[g; phi].
    ricci_sign=+1:  4 R_{mrns} f^{rs} - 2 R f_mn + 2 g R box f - 4 R_mn box f + 4 R_m^r f_rn + 4 R_n^r f_rm - 4 g R^rs f_rs
    ricci_sign=-1:  the six Ricci-dependent terms with the opposite signs (manuscript Eq. A6 as of 2026-10-05)."""
    gi=G["gi"]; H,_=hess_lower(bg,G,c)
    Huu=np.einsum("...ma,...nb,...ab->...mn",gi,gi,H)
    box=np.einsum("...mn,...mn->...",gi,H)
    Ric=G["Ric"]; Rs=G["Rs"]
    Ric_ud=np.einsum("...ma,...ar->...mr",Ric,gi)                 # R_mu^rho
    ric=(-2*Rs[...,None,None]*H+2*gd*(Rs*box)[...,None,None]-4*Ric*box[...,None,None]
         +4*np.einsum("...mr,...rn->...mn",Ric_ud,H)+4*np.einsum("...nr,...rm->...mn",Ric_ud,H)
         -4*gd*np.einsum("...rs,...rs->...",np.einsum("...ra,...sb,...ab->...rs",gi,gi,Ric),H)[...,None,None])
    return 4*np.einsum("...mrns,...rs->...mn",G["Rd"],Huu)+ricci_sign*ric

def Hdd(bg,G,gd,c):
    """H_{mu nu} = -4 (**R)_{mu rho nu sigma} nabla^rho nabla^sigma phi, built directly from the double dual
    (**R)^{ab}_{cd} = 1/4 eps^{ab}_{ef} R^{ef}_{gh} eps^{gh}_{cd}; no term-by-term signs enter."""
    gi=G["gi"]; H,_=hess_lower(bg,G,c)
    Huu=np.einsum("...ma,...nb,...ab->...mn",gi,gi,H)
    dd_uudd=0.5*np.einsum("...abef,...efcd->...abcd",G["dual_uudd"],G["euudd"])
    dd=np.einsum("...ai,...bj,...ijcd->...abcd",gd,gd,dd_uudd)
    return -4*np.einsum("...mrns,...rs->...mn",dd,Huu)

def Hfull(bg,G,gd,c):
    return Hterms(bg,G,gd,c,+1.0)

def Cfull_dualterm(bg,G,c):
    """-4 sym[(nabla^a nabla^d theta) *R_{a mu nu d}] on a general metric (dual-Riemann term only)."""
    gi=G["gi"]; H,_=hess_lower(bg,G,c)
    Huu=np.einsum("...ma,...nb,...ab->...mn",gi,gi,H)
    C=np.einsum("...ad,...amnd->...mn",Huu,G["dual_dddd"])
    return -4*0.5*(C+np.swapaxes(C,-1,-2))

def Tfull(bg,G,gd,c):
    _,grad=hess_lower(bg,G,c)
    sq=np.einsum("...mn,...m,...n->...",G["gi"],grad,grad)
    return np.einsum("...m,...n->...mn",grad,grad)-0.5*gd*sq[...,None,None]

def Gfull(G,gd):
    return G["Ric"]-0.5*gd*G["Rs"][...,None,None]

def main(spin=0.6,N=20,eps=1e-5,seed=3):
    g=SpectralGrid(Nz=N,Nchi=N,spin=spin); bg=KerrBackground(g)
    # use a bulk mask well away from horizon/poles/infinity to keep FD round-off down
    m=np.zeros(bg.shape,bool); m[4:-6,4:-4]=True
    rng=np.random.default_rng(seed)
    def synthH(scale):
        H={}
        for i in (1,2,3,4):
            C=np.zeros((N+1,N+1))
            for l in (0,2,4):
                C[1:5,l]=rng.standard_normal(4)*0.4**np.arange(4)*scale
            C[0,:]=0
            # make it vanish at infinity (z=-1): subtract value there
            for l in (0,2,4):
                C[0,l]=-sum(C[n,l]*(-1)**n for n in range(1,5))
            H[i]=C
        return H
    Ha=synthH(1.0); Hb=synthH(1.0)
    def synthS(par):
        C=np.zeros((N+1,N+1)); ls=(0,2,4) if par=="even" else (1,3)
        for l in ls:
            C[1:5,l]=rng.standard_normal(4)*0.5**np.arange(4)
            C[0,l]=-sum(C[n,l]*(-1)**n for n in range(1,5))
        return C
    phi=synthS("even"); th=synthS("odd")
    h,dh,d2h=assemble_h(bg,Hb)      # "opposite sector" metric perturbation
    d2g0=_bg_d2g(bg)
    def G_at(e):
        with np.errstate(all="ignore"):
            return geom(bg,bg.g_dd+e*h,bg.dg_ddd+e*dh,d2g0+e*d2h), bg.g_dd+e*h
    Gp,gp=G_at(+eps); Gm,gm=G_at(-eps); G0,g0=G_at(0.0)
    fd=lambda f:(f(Gp,gp)-f(Gm,gm))/(2*eps)
    def rel(a,b):
        s=np.nanmax(np.abs(b[m])); return np.nanmax(np.abs((a-b)[m]))/s
    htr=np.einsum("...mn,...mn->...",bg.g_uu,h)
    print(f"spin={spin} N={N} eps={eps}   (rel = max|code - brute force| / max|brute force|, bulk nodes)")
    print("-- sanity: brute-force machinery at eps=0 --")
    print("   GB(g0) vs closed form      :",rel(GB(G0),bg.gb_scalar))
    print("   Pont(g0) vs closed form    :",rel(PONT(G0),bg.pontryagin))
    print("   Hfull(g0) vs H_sGB_leading :",rel(Hfull(bg,G0,g0,phi),H_sGB_leading(bg,phi)))
    print("   Cdual(g0) vs C_dCS_leading :",rel(Cfull_dualterm(bg,G0,th),C_dCS_leading(bg,th)))
    print("   dG/deps vs G1_lin[h]       :",rel(G1_lin(bg,h,dh,d2h),fd(lambda G,gd:Gfull(G,gd))))
    print("-- Stage-I scalar sources --")
    bf=fd(lambda G,gd:GB(G)); code=gb_invariant_pert(bg,h,dh,d2h)
    print("   GB^(1):   code                         rel",rel(code,bf))
    print("   GB^(1):   code - 1/2 h GB^(0)          rel",rel(code-0.5*htr*bg.gb_scalar,bf))
    bf=fd(lambda G,gd:PONT(G)); code=pontryagin_pert(bg,h,dh,d2h)
    print("   *RR^(1):  code                         rel",rel(code,bf))
    bf=fd(lambda G,gd:BOX(bg,G,phi)); code=box_pert(bg,h,dh,phi)
    print("   box^(1):  code                         rel",rel(code,bf))
    print("-- Stage-II source pieces (terms linear in the opposite-sector h) --")
    zero=np.zeros((N+1,N+1))
    bf=fd(lambda G,gd:Hfull(bg,G,gd,phi)); code=H_mixed(bg,phi,zero,h,dh,d2h)
    print("   H^(1,1) source part: code              rel",rel(code,bf))
    bdd=fd(lambda G,gd:Hdd(bg,G,gd,phi))
    print("   sign of the Ricci terms, against -4 (**R) nabla nabla phi built from the double dual:")
    print("      seven-term form, literature signs    rel",rel(bf,bdd))
    print("      seven-term form, manuscript (A6)     rel",rel(fd(lambda G,gd:Hterms(bg,G,gd,phi,-1.0)),bdd))
    # decomposition of what is missing
    huu=np.einsum("...ma,...nb,...ab->...mn",bg.g_uu,bg.g_uu,h)
    hess=covariant_hessian(bg,phi)
    hess_ud=np.einsum("...ab,...bs->...as",hess,bg.g_uu)              # nabla_a nabla^s phi
    raise_fix=-4*(np.einsum("...mrns,...ra,...as->...mn",bg.Riemann_dddd,huu,hess_ud)
                  +np.einsum("...mrns,...sa,...ar->...mn",bg.Riemann_dddd,huu,hess_ud))
    print("   H^(1,1): code + index-raising terms    rel",rel(code+raise_fix,bf))
    # Ricci terms: linearised Ricci of h on Kerr
    dRic=(Gp["Ric"]-Gm["Ric"])/(2*eps); dRs=np.einsum("...ab,...ab->...",bg.g_uu,dRic)
    box0=np.einsum("...mn,...mn->...",bg.g_uu,hess)
    dRic_ud=np.einsum("...ma,...ar->...mr",dRic,bg.g_uu)
    ricci_fix=(-2*dRs[...,None,None]*hess+2*bg.g_dd*(dRs*box0)[...,None,None]-4*dRic*box0[...,None,None]
               +4*np.einsum("...mr,...rn->...mn",dRic_ud,hess)+4*np.einsum("...nr,...rm->...mn",dRic_ud,hess)
               -4*bg.g_dd*np.einsum("...rs,...rs->...",np.einsum("...ra,...sb,...ab->...rs",bg.g_uu,bg.g_uu,dRic),hess)[...,None,None])
    print("   H^(1,1): code + raising + Ricci terms  rel",rel(code+raise_fix+ricci_fix,bf))
    bf=fd(lambda G,gd:Cfull_dualterm(bg,G,th)); code=C_mixed(bg,th,zero,h,dh,d2h)
    print("   C^(1,1) dual-Riemann source part: code rel",rel(code,bf))
    hessT=covariant_hessian(bg,th)
    dual=np.einsum("...ai,...bj,...ijcd->...abcd",bg.g_dd,bg.g_dd,bg.dualRiemann_uudd)
    X=-np.einsum("...ab,...bd,...amnd->...mn",huu,np.einsum("...bs,...sd->...bd",hessT,bg.g_uu),dual) \
      -np.einsum("...db,...ba,...amnd->...mn",huu,np.einsum("...bs,...sa->...ba",hessT,bg.g_uu),dual)
    craise=-4*0.5*(X+np.swapaxes(X,-1,-2))
    print("   C^(1,1): code + index-raising terms    rel",rel(code+craise,bf))
    bf=fd(lambda G,gd:Tfull(bg,G,gd,phi))      # metric dependence of T[phi0] at fixed phi0
    print("   T^(1,1): metric-perturbation part of T[phi0]; code has 0.  max|missing| / max|T[phi0]| =",
          np.nanmax(np.abs(bf[m]))/np.nanmax(np.abs(Tfull(bg,G0,g0,phi)[m])), "(per unit h)")
    # G^(2): second mixed difference of the exact Einstein tensor
    ha,dha,d2ha=assemble_h(bg,Ha)
    def Gab(ea,eb):
        with np.errstate(all="ignore"):
            gd=bg.g_dd+ea*ha+eb*h
            return Gfull(geom(bg,gd,bg.dg_ddd+ea*dha+eb*dh,d2g0+ea*d2ha+eb*d2h),gd)
    e=1e-3
    bf=(Gab(e,e)-Gab(e,-e)-Gab(-e,e)+Gab(-e,-e))/(4*e*e)
    code=2*G2_bilinear(bg,(ha,dha,d2ha),(h,dh,d2h))
    print("   2 G^(2)[ha,hb] vs exact mixed difference rel",rel(code,bf))
if __name__=="__main__":
    main()
