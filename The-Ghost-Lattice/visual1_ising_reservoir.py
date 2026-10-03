# visual1_ising_reservoir.py
import numpy as np
import plotly.graph_objects as go

np.random.seed(42)
N = 29

# FC-TFI couplings
J = np.random.randn(N, N) * 0.15
J = (J + J.T) / 2
np.fill_diagonal(J, 0)
h = np.random.randn(N) * 0.3
W_in = np.random.randn(N) * 0.6

# Fibonacci sphere layout
def fib_sphere(n):
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)

x, y, z = fib_sphere(N)

# Simulate mean-field dynamics
T_steps = 240
states = np.zeros((T_steps, N))
input_sig = np.sin(np.linspace(0, 8*np.pi, T_steps))
s = np.random.randn(N) * 0.05
for t in range(T_steps):
    s = np.tanh(J @ s + h + W_in * input_sig[t])
    states[t] = s

# Top couplings only (avoid hairball)
thr = np.percentile(np.abs(J[np.triu_indices(N, 1)]), 85)
edges_x, edges_y, edges_z = [], [], []
for i in range(N):
    for j in range(i+1, N):
        if abs(J[i, j]) > thr:
            edges_x += [x[i], x[j], None]
            edges_y += [y[i], y[j], None]
            edges_z += [z[i], z[j], None]

edge_trace = go.Scatter3d(
    x=edges_x, y=edges_y, z=edges_z,
    mode='lines',
    line=dict(color='rgba(90,150,220,0.35)', width=2),
    hoverinfo='none', showlegend=False
)

frames = []
for t in range(T_steps):
    inten = np.abs(states[t])
    node = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=7 + 14*inten,
            color=states[t],
            colorscale='RdBu', cmin=-1, cmax=1,
            colorbar=dict(title='⟨σᶻ⟩', x=1.02)
        ),
        name=f'Qubits', showlegend=False
    )
    frames.append(go.Frame(data=[edge_trace, node], name=str(t)))

fig = go.Figure(data=[edge_trace, frames[0].data[1]], frames=frames)
fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
               zaxis=dict(visible=False), bgcolor='#05070d'),
    paper_bgcolor='#05070d',
    title=dict(text='Ghost-Lattice  ·  29-Qubit FC-TFI Reservoir',
               font=dict(color='white', size=18)),
    updatemenus=[dict(
        type='buttons', x=0.05, y=1.1, xanchor='left',
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=45, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ]
    )]
)
fig.show()