
# 20220419
# plotly 축 이름 변경, font size 변경, x axis, axiss
import plotly.express as px

fig = px.violin(corr_save, y='Correlation coefficient', box=True, points='all', color='pain state')
# Correlation coefficient between calcicum activity and movements
fig.update_layout(
    title="",
    xaxis_title="",
    yaxis_title="Pearson's r",
    legend_title="Pain state",
    font=dict(
        family="Courier New, monospace",
        size=20,
        color="RebeccaPurple"
    )
)
fig.show()

if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image("images/fig1.png", scale=10)
