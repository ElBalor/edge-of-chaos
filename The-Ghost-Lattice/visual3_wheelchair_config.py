# visual3_wheelchair_config.py
import numpy as np
import plotly.graph_objects as go

np.random.seed(3)

# Floor
gx, gy = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
gz = np.zeros_like(gx)

# Obstacle (singularity)
obs_x, obs_y = 0.6, 0.0
obs_r = 0.7
field = -np.exp(-((gx - obs_x)**2 + (gy - obs_y)**2) / (2 * 0.4**2))
# Wells around start and goal
field += 0.6 * np.exp(-((gx + 2.2)**2 + (gy + 1.2)**2) / (2 * 0.5**2))
field += 0.6 * np.exp(-((gx - 2.2)**2 + (gy - 1.2)**2) / (2 * 0.5**2))

field_surf = go.Surface(
    x=gx, y=gy, z=field * 0.15,
    colorscale=[[0, '#0a1a2f'], [0.5, '#1e4d7a'],
                [0.8, '#4a90c2'], [1.0, '#a8d8ff']],
    opacity=0.35, showscale=False, name='AlienX field'
)

# Obstacle marker
u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
ox = obs_x + obs_r * np.cos(u) * np.sin(v)
oy = obs_y + obs_r * np.sin(u) * np.sin(v)
oz = 0.35 + 0.35 * np.cos(v)
obstacle = go.Surface(
    x=ox, y=oy, z=oz,
    colorscale=[[0, '#6b1010'], [1, '#ff4444']],
    opacity=0.85, showscale=False, name='Singularity'
)

# Wheelchair path: advection around obstacle (simple spline avoidance)
t = np.linspace(0, 1, 80)
def path(t):
    x = -2.2 + 4.4 * t
    y = -1.2 + 2.4 * t + 1.4 * np.sin(np.pi * t)  # detour
    return x, y, np.zeros_like(x)

px, py, pz = path(t)

# Frames: chair moves, trail grows
frames = []
for k in range(1, len(t) + 1):
    # trail up to k
    trail = go.Scatter3d(
        x=px[:k], y=py[:k], z=pz[:k] + 0.05,
        mode='lines',
        line=dict(color='#00ffc8', width=6),
        name='trajectory', showlegend=False
    )
    # chair as a box-ish marker
    chair = go.Scatter3d(
        x=[px[k-1]], y=[py[k-1]], z=[pz[k-1] + 0.15],
        mode='markers',
        marker=dict(size=18, color='#00ffc8', symbol='square',
                    line=dict(color='white', width=2)),
        name='wheelchair', showlegend=False
    )
    goal = go.Scatter3d(
        x=[2.2], y=[1.2], z=[0.15],
        mode='markers',
        marker=dict(size=16, color='#a8ff60', symbol='diamond',
                    line=dict(color='white', width=2)),
        name='goal', showlegend=False
    )
    frames.append(go.Frame(data=[field_surf, obstacle, trail, chair, goal],
                           name=str(k)))

fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        zaxis=dict(visible=False), bgcolor='#05070d',
        aspectmode='data',
        camera=dict(eye=dict(x=0.0, y=-1.6, z=1.4))
    ),
    paper_bgcolor='#05070d',
    title=dict(text='AlienX Configuration Manifold  ·  −∇‖Ψ‖² + λT_stress',
               font=dict(color='white', size=18)),
    updatemenus=[dict(
        type='buttons', x=0.05, y=1.1, xanchor='left',
        buttons=[
            dict(label='▶ Drive', method='animate',
                 args=[None, dict(frame=dict(duration=70, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ]
    )]
)
fig.show()