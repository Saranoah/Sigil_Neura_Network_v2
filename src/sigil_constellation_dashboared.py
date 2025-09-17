import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import base64
import os

def build_constellation_graph(layer_infos):
    # Build NetworkX graph
    G = nx.Graph()
    for info in layer_infos:
        G.add_node(info['layer_name'], archetype=info['archetype'],
                   sigil_img_path=info['sigil_img_path'], metrics=info['metrics'])
        for nbr in info.get('neighbors', []):
            G.add_edge(info['layer_name'], nbr)
    return G

def encode_image(image_path):
    # Encode image to display as node
    if not os.path.exists(image_path):
        return None
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + encoded

def make_plot(G):
    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(len(G.nodes)))
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='white'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_images = []
    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        info = G.nodes[node]
        m = info['metrics']
        metrics_str = (f"{node}\n{info['archetype']}\n"
                       f"L:{m['L_l']:.2f} FD:{m['FD_l']:.2f} H:{m['H_l']:.2f} ρ:{m['ρ_l']:.2f}")
        node_text.append(metrics_str)
        img_url = encode_image(info['sigil_img_path'])
        node_images.append(img_url)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[f"<b>{t.splitlines()[0]}</b>" for t in node_text],
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=70,
            color='gold',
            symbol='circle',
            opacity=0.7,
            line=dict(width=4, color='white'),
        ),
        textposition="bottom center"
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<b>Sigil Network Constellation</b>',
                        titlefont=dict(size=24, color='gold'),
                        showlegend=False,
                        hovermode='closest',
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        margin=dict(b=60,l=60,r=60,t=80),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    # Overlay sigil images as custom annotations
    for i, img_url in enumerate(node_images):
        if img_url:
            fig.add_layout_image(
                dict(
                    source=img_url,
                    x=node_x[i]-0.09, y=node_y[i]+0.09,
                    sizex=0.18, sizey=0.18,
                    xref="x", yref="y",
                    layer="above"
                )
            )
    return fig

# Example layer_infos, replace with real ones
layer_infos = [
    {
        "layer_name": "Encoder",
        "archetype": "The Oracle",
        "sigil_img_path": "sigil_gallery/Encoder_TheOracle_epoch_0020_20250911_230215.png",
        "metrics": {"L_l": 0.34, "FD_l": 0.22, "H_l": 0.92, "ρ_l": 1.23},
        "neighbors": ["Midlayer"]
    },
    {
        "layer_name": "Midlayer",
        "archetype": "The Alchemist",
        "sigil_img_path": "sigil_gallery/Midlayer_TheAlchemist_epoch_0020_20250911_230215.png",
        "metrics": {"L_l": 0.51, "FD_l": 0.60, "H_l": 0.80, "ρ_l": 1.01},
        "neighbors": ["Encoder", "Classifier"]
    },
    {
        "layer_name": "Classifier",
        "archetype": "The Sentinel",
        "sigil_img_path": "sigil_gallery/Classifier_TheSentinel_epoch_0020_20250911_230215.png",
        "metrics": {"L_l": 0.77, "FD_l": 0.10, "H_l": 0.35, "ρ_l": 1.75},
        "neighbors": ["Midlayer"]
    }
]

G = build_constellation_graph(layer_infos)

app = dash.Dash(__name__)
app.title = "Sigil Network Constellation"

app.layout = html.Div([
    html.H1("Sigil Network Constellation", style={'color': 'gold', 'text-align': 'center'}),
    dcc.Graph(
        id='constellation-graph',
        figure=make_plot(G),
        style={'height': '900px', 'backgroundColor': 'black'}
    ),
    html.Div(id='node-info', style={'color': 'white', 'font-size': '18px', 'margin-top': '24px', 'text-align': 'center'})
])

@app.callback(
    Output('node-info', 'children'),
    Input('constellation-graph', 'clickData')
)
def display_node_info(clickData):
    if clickData and 'points' in clickData:
        pt = clickData['points'][0]
        text = pt['hovertext']
        lines = text.split('\n')
        return html.Div([
            html.H3(lines[0], style={'color':'gold'}),
            html.P(f"Archetype: {lines[1]}", style={'color':'#8ecae6'}),
            html.P(f"Metrics: {lines[2]}", style={'color':'#ffb703'}),
            html.P(f"More: {lines[3]}", style={'color':'#ffb703'}) if len(lines)>3 else ""
        ])
    return "Click a node to see its mythic metrics and archetype."

if __name__ == '__main__':
    app.run_server(debug=True)
