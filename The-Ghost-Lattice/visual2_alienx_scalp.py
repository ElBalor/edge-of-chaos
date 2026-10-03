# visual2_alienx_scalp.py
import numpy as np
import plotly.graph_objects as go

np.random.seed(7)

# Approximate 10-20 electrode layout on an ellipsoid head
# (azimuth, elevation in degrees)
positions_deg = [
    (0, 80), (36, 65), (-36, 65), (72, 40), (-72, 40),
    (108, 65), (-108, 65), (144, 80), (-144, 80),
    (0, 20), (45, 10), (-45, 10), (90, 20), (-90, 20),
    (135, 10), (-135, 10), (180, 20),
    (22, 45), (-22, 45), (0, -10), (60, -10), (-60, -10)
]
N = len(positions_deg)

a, b, c = 1.0, 1.15, 1.3   # head is taller than wide
def head_xyz(az_deg, el_deg):
    az = np.radians(az_deg); el = np.radians(el_deg)
    x = a * np.cos(el) * np.cos(az)
    y = b * np.cos(el) * np.sin(az)
    z = c * np.sin(el)
    return x, y, z

pos = np.array([head_xyz(*p) for p in positions_deg])

# σ_i from local neighbor distance
sigma = np.zeros(N)
for i in range(N):
    d = np.linalg.norm(pos - pos[i], axis=1)
    d[i] = np.inf
    sigma[i] = np.sort(d)[:3].mean() * 0.6

# Simulate an EMG artifact on electrode 10 (a frontal one) at t=0.5
artifact_electrode = 10
artifact_phase = np.linspace(0, 1, 60)

frames = []
for t_idx, p in enumerate(artifact_phase):
    spike = np.exp(-((p - 0.5)**2) / 0.005)
    intensity = np.ones(N) * 0.3
    intensity[artifact_electrode] = 0.3 + 0.7 * spike
    # Neighbors of the attacked node heat up too (topological pinch)
    dists = np.linalg.norm(pos - pos[artifact_electrode], axis=1)
    neighbor_heat = np.exp(-dists / 0.5) * spike
    intensity = np.clip(intensity + 0.5 * neighbor_heat, 0, 1)

    node_trace = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(
            size=8 + 18 * intensity,
            color=intensity,
            colorscale=[[0, '#1a2b4c'], [0.4, '#2d4a7a'],
                        [0.7, '#f4a261'], [1.0, '#c1121f']],
            cmin=0, cmax=1,
            colorbar=dict(title='thermal<br>stress', x=1.02)
        ),
        name='Electrodes', showlegend=False
    )
    # Local frame arrows at each electrode (short, colored by stress)
    ax_x, ax_y, ax_z = [], [], []
    for i in range(N):
        n = pos[i] / np.linalg.norm(pos[i])
        # tangent frame
        up = np.array([0, 0, 1.0])
        t1 = np.cross(n, up); t1 /= (np.linalg.norm(t1) + 1e-8)
        t2 = np.cross(n, t1)
        s = 0.12 * (1 + 0.5*intensity[i])
        ax_x += [pos[i,0], pos[i,0]+s*t1[0], None,
                 pos[i,0], pos[i,0]+s*t2[0], None]
        ax_y += [pos[i,1], pos[i,1]+s*t1[1], None,
                 pos[i,1], pos[i,1]+s*t2[1], None]
        ax_z += [pos[i,2], pos[i,2]+s*t1[2], None,
                 pos[i,2], pos[i,2]+s*t2[2], None]
    frame_trace = go.Scatter3d(
        x=ax_x, y=ax_y, z=ax_z, mode='lines',
        line=dict(color='rgba(120,200,255,0.55)', width=1.5),
        hoverinfo='none', showlegend=False
    )
    frames.append(go.Frame(data=[node_trace, frame_trace], name=str(t_idx)))

# Initial frame
init = frames[0].data
fig = go.Figure(data=init, frames=frames)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        zaxis=dict(visible=False), bgcolor='#05070d',
        aspectmode='data'
    ),
    paper_bgcolor='#05070d',
    title=dict(text='AlienX Scalp Manifold  ·  22 Electrodes, σ-normalized',
               font=dict(color='white', size=18)),
    updatemenus=[dict(
        type='buttons', x=0.05, y=1.1, xanchor='left',
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=60, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ]
    )]
)
fig.show()