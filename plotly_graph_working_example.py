from plotly.offline import iplot
import plotly.graph_objs as go
# Plotly version 4.0.0 (pip install plotly==4.0.0)

trace0 = go.Scatter(x=[0, 1, 2, 3, 4],
                    y=[0, 1, 4, 9, 16], mode='lines+markers',  # Select 'lines', 'markers' or 'lines+markers'
                    name='legend --> x**2')
trace1 = go.Scatter(x=[0, 1, 2, 3, 4],
                    y=[10, 8, 7, 4, 1], mode='lines+markers',
                    name='legend --> -x')

data = [trace0, trace1]

layout = {'title': 'Chart title --> x**2',
          'xaxis': {'title': 'xlabel --> x [m]',
                    'type': 'linear'},  # Select 'log' or 'linear'
          'yaxis': {'title': 'ylabel --> -x',
                    'type': 'linear'},  # Select 'log' or 'linear'
          'template': 'plotly_dark'}

iplot({'data': data, 'layout': layout})
