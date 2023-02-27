import numpy as np
from dash import dcc, html
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots
from torch_geometric.utils import to_networkx


MANEUVERS = ['start', 'takeoff', 'turn', 'line', 'orbit', 'landing', 'stop']


def plot_2d(data, lim=10):
    # The graph to visualize
    G = to_networkx(data)
    pos = data.pos.numpy()

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v, :2] for v in sorted(G)])
    edge_xyz = np.array([(pos[u, :2], pos[v, :2]) for u, v in G.edges()])

    # Create the 2D figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, c=data.token.numpy(), cmap="rainbow")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    # Turn gridlines off
    # ax.grid(False)
        
    # Suppress tick labels
    # for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #     dim.set_ticks([])
        
    # Set axes labels and limits
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal', 'box')

    # fig.tight_layout()
    plt.show()


def plot_3d(data, lim=10):
    # The graph to visualize
    G = to_networkx(data)
    pos = data.pos.numpy()

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, c=data.token.numpy(), cmap="rainbow")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    # Turn gridlines off
    # ax.grid(False)
        
    # Suppress tick labels
    # for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #     dim.set_ticks([])
        
    # Set axes labels and limits
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # fig.tight_layout()
    plt.show()


def plotly_2d(true_df, pred_df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("True Maneuver", "Pred Maneuver"))
    fig.add_trace(go.Scatter(x=true_df['x'], y=true_df['y'], showlegend=False, visible=False), row=1,col=1)
    fig.add_trace(go.Scatter(x=true_df['x'], y=true_df['y'], showlegend=False, visible=False), row=1,col=2)
    fig.update_traces(mode='markers', marker=dict(colorscale=[[0, 'rgb(150, 150, 250)'], [1, 'rgb(250, 0, 0)']], showscale=True))
    for s, c in zip(MANEUVERS, px.colors.qualitative.Plotly):
        t_mask, p_mask = true_df['maneuver'] == s, pred_df['maneuver'] == s   
        fig.add_trace(go.Scatter(x=true_df['x'][t_mask], y=true_df['y'][t_mask], name=s, mode='markers', marker_color=c, showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=true_df['x'][p_mask], y=true_df['y'][p_mask], name=s, mode='markers', marker_color=c, showlegend=True), row=1,col=2)
    c = [s.marker.color for s in fig.data[2:]]
    dropdown_menu = dict(buttons=list([
            dict(label="All Maneuvers", method="update",  args=[{"visible": [False, False] + [True]*10, 'marker.color':[None,None] + c}]),
              ] + [dict(label=k, method="update",  args=[{"visible": [True, True] + [False] * 10, "marker.color": [(true_df['maneuver']==k).astype(float), pred_df[k]]}]) for k in MANEUVERS]
        ),  direction="down", showactive=True, xanchor="right", yanchor="top")
    fig.update_layout(updatemenus=[dropdown_menu], legend=dict(yanchor="bottom", y=0.30, xanchor="left", x=0))
    fig.show()

