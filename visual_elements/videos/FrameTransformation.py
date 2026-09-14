import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

NFRAMES, FPS = 140, 20
TRAIL_FRAMES = 28
LAB_ALPHA = 0.14
GHOST_ALPHA = 0.05
TRANSFORMED_ALPHA = 0.24
ORANGE = "#E68613"
OUT = "bloch_three_frames_v8.gif"

def Ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def sphere():
    C=[]
    for z in np.linspace(-.75,.75,7):
        a=np.linspace(0,2*np.pi,90); r=np.sqrt(1-z*z)
        C.append(np.c_[r*np.cos(a), r*np.sin(a), 0*a+z])
    th=np.linspace(0,np.pi,90)
    for ph in np.linspace(0,2*np.pi,12,endpoint=False):
        C.append(np.c_[np.sin(th)*np.cos(ph),
                       np.sin(th)*np.sin(ph),
                       np.cos(th)])
    return C

S = sphere()
u = np.linspace(0,2*np.pi,NFRAMES,endpoint=False)

# large radial motion, then smaller, then larger again
env = 0.90*(0.62 + 0.38*np.cos(u))
rho = env*np.cos(3*u)
rho = np.clip(rho, -0.92, 0.92)

# smooth moving frame, with R(0)=I
Rs = np.array([Rz(1.18*q+0.22*np.sin(q)) @
               Ry(0.58*np.sin(0.85*q)) @
               Rz(0.22*np.sin(1.30*q)) for q in u])

rr = np.c_[np.zeros_like(rho), np.zeros_like(rho), rho]
rl = np.einsum("nij,nj->ni", Rs, rr)

def Eframe_one(P, r):
    eps = 1e-12
    Q = P.copy()
    if r >= 0:
        lam = 1.0/max(1.0-r, eps)
        Q[:,0] *= np.sqrt(lam)
        Q[:,1] *= np.sqrt(lam)
        Q[:,2]  = lam*Q[:,2] + (1.0-lam)
        label = r"active: $-\mathcal{D}[\sigma^-]$"
    else:
        lam = 1.0/max(1.0+r, eps)
        Q[:,0] *= np.sqrt(lam)
        Q[:,1] *= np.sqrt(lam)
        Q[:,2]  = lam*Q[:,2] + (lam-1.0)
        label = r"active: $-\mathcal{D}[\sigma^+]$"
    return Q, label, lam

def set3(L,P):
    L.set_data(P[:,0], P[:,1]); L.set_3d_properties(P[:,2])

dummy = np.array([[[0.,0.,0.],[0.,0.,0.]]])
def trail(C,P,i):
    q = P[max(0,i-TRAIL_FRAMES+1):i+1]
    if len(q) < 2:
        C.set_segments(dummy); C.set_color([(0,0,0,0)]); return
    seg = np.stack([q[:-1], q[1:]], 1)
    a = np.linspace(.04,.95,len(seg))
    rgb = np.array([230,134,19])/255
    C.set_segments(seg)
    C.set_color(np.c_[np.tile(rgb,(len(seg),1)), a])

fig=plt.figure(figsize=(11.0,3.85))
ax=[fig.add_subplot(1,3,k+1,projection="3d") for k in range(3)]
titles=["(1) Lab frame", "(2) Rotating frame", "(3) Co-moving frame"]
for A,t in zip(ax, titles):
    A.set(xlim=(-1.08,1.08), ylim=(-1.08,1.08), zlim=(-1.08,1.08))
    A.set_box_aspect((1,1,1)); A.set_axis_off(); A.view_init(20,35); A.set_title(t)

for c in S:
    ax[0].plot(*c.T, color=".30", lw=.65, alpha=LAB_ALPHA)
    ax[1].plot(*c.T, color=".25", lw=.55, alpha=GHOST_ALPHA)
    ax[2].plot(*c.T, color=".22", lw=.55, alpha=GHOST_ALPHA)

g2=[ax[1].plot([],[],[], color=".30", lw=.65, alpha=LAB_ALPHA)[0] for _ in S]
g3=[ax[2].plot([],[],[], color=".28", lw=.95, alpha=TRANSFORMED_ALPHA)[0] for _ in S]
cols=["#B23A48","#3A7D44","#3A5FCD"]
a2=[ax[1].plot([],[],[], lw=1.2, alpha=.45, color=c)[0] for c in cols]
p=[A.plot([],[],[],"o", ms=6.6, color=ORANGE)[0] for A in ax]
tr=[Line3DCollection(dummy, linewidths=2.3) for _ in range(2)]
ax[0].add_collection3d(tr[0], autolim=False)
ax[1].add_collection3d(tr[1], autolim=False)
txt=ax[2].text2D(0.03,0.97,"", transform=ax[2].transAxes, va="top", ha="left",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=1.8))

def animate(i):
    R = Rs[i]

    p[0].set_data([rl[i,0]], [rl[i,1]])
    p[0].set_3d_properties([rl[i,2]])
    trail(tr[0], rl, i)

    for L,c in zip(g2,S): set3(L, c@R)
    for k,L in enumerate(a2):
        v=R.T[:,k]; set3(L, np.vstack((-v,v)))
    p[1].set_data([0],[0]); p[1].set_3d_properties([rho[i]])
    trail(tr[1], rr, i)

    label=""; lam=1.0
    for L,c in zip(g3,S):
        Q,label,lam = Eframe_one(c.copy(), rho[i])
        set3(L, Q)
    p[2].set_data([0],[0]); p[2].set_3d_properties([0])
    txt.set_text(label + rf"\n$\lambda={lam:.2f}$")

ani=FuncAnimation(fig, animate, frames=NFRAMES, interval=1000/FPS, blit=False)
fig.tight_layout(pad=.6)
ani.save(OUT, writer=PillowWriter(fps=FPS), dpi=125)