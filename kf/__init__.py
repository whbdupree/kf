from jax import numpy as jnp
from jax.lax import fori_loop, cond
from jax import vmap

dt = 0.1

gSynE = 10.
gSynI = 60.
ESynE = 0.
ESynI = -75.

gNaP = 5.
gK = 5.
gAD = 10.
gL = 2.8
ENa = 50.
EK = -85.
EL = jnp.array([-60.,-60.,-60.,-60.,-64.])

ThNaP_max= 4000.
TmAD = 2000.

VmNaP = -40.
kmNaP = -6.
VhNaP = -55.
khNaP = 10.
VmK = -30.
kmK = -4.

C = 20.

def mNaP(vv):
    return 1./( 1. + jnp.exp( (vv-VmNaP)/kmNaP ) )

def hNaP(vv):
    return 1./( 1. + jnp.exp( (vv-VhNaP)/khNaP ) )

def mK(vv):
    return 1./( 1. + jnp.exp( (vv-VmK)/kmK ) )

def ThNaP(vv):
    return ThNaP_max/jnp.cosh( (vv-VhNaP)/khNaP )

Vmax = -20
Vmin = -50

def f(V):
    x = cond( V > Vmin,
              lambda V: (V - Vmin) / (Vmax - Vmin),
              lambda _:0., V )
    x = cond( V < Vmax, lambda _:x, lambda _:1., V )
    return x

vmapf = vmap(f)

def NaP_cell( vv, hh ):
    INaP = gNaP * mNaP(vv) * hh * ( vv - ENa )
    return INaP


def AD_cell( vv, mm ):
    IAD = gAD * mm * ( vv - EK )
    mkv = mK(vv)
    return IAD




def step( i, state ):
    V,m = state
    r = vmapf(V)
    ISynE = gSynE * ( jnp.dot( a , r ) + d ) * ( V - ESynE )
    ISynI = gSynI * jnp.dot( b , r ) * ( V - ESynI )

    mkv = mK(V)
    IK = gK * mkv * mkv* mkv* mkv * ( V - EK )
    IL = gL * ( V - EL )
    ICell = jnp.array([
        NaP_cell(V[0], m[0]),
        AD_cell(V[1], m[1]),
        0,
        AD_cell(V[3], m[3]),
        NaP_cell(V[4], m[4])
    ]) + IK + IL + ISynE + ISynI

    dmdt = jnp.array([
        ( hNaP(V[0]) - m[0] ) / ThNaP(V[0]),
        ( f(V[1]) - m[1] ) / TmAD,
        ( f(V[2]) - m[2] ) / TmAD,
        ( f(V[3]) - m[3] ) / TmAD,
        ( hNaP(V[4]) - m[4] ) / ThNaP(V[4]),
    ])
    
    return [V + dt*(-ICell/C), m + dt*dmdt]

def euler10( i, state ):
    return fori_loop(0,10, step, state)
    #for i in range(2):
    #    state = step(i,state)
    
a = jnp.array([
    [ 0., 0.,   0., 0., 0.5],
    [ 0.35, 0., 0., 0. ,0. ],
    [ 0., 0.,   0., 0., 0.25],
    [ 0. ]*5,
    [ 0. ]*5
])

b = jnp.array([
    [0.,   0.,   0.15, 1.,   0.  ],
    [0.,   0.,   0.15, 0.42, 0.  ],
    [0.,   0.42, 0.,   0.2,  0.  ],
    [0.,   0.22, 0.,   0.,   0.  ],
    [0.,   0.075, 0.,  0.12, 0.   ]
]) 

d = jnp.array([ 0.8, 0.9, 1, .7, 0.15*1.5])
