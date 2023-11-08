# TODO: Develop this by adding further plot functions.

# def get_distributions_figure(feature_names, frame1, frame2, /, **kwargs):
#     histnorm = kwargs.get("histnorm", "probability density")
#     train_color = kwargs.get("train_color", "blue")
#     test_color = kwargs.get("test_color", "red")
#     n_cols = kwargs.get("n_cols", 3)
#     n_rows, axes = get_n_rows_and_axes(len(feature_names), n_cols)

#     fig = make_subplots(
#         rows=n_rows,
#         cols=n_cols,
#         y_title=histnorm.title(),
#         horizontal_spacing=kwargs.get("horizontal_spacing", 0.1),
#         vertical_spacing=kwargs.get("vertical_spacing", 0.1),
#     )
#     fig.update_annotations(font_size=kwargs.get("annotations_font_size", 14))

#     for frame, color, name in zip(
#         (frame1, frame2), (train_color, test_color), ("Train", "Test")
#     ):
#         if frame is None:  # Test dataset may not exist.
#             break

#         for k, (var, (row, col)) in enumerate(zip(feature_names, axes), start=1):
#             # density, bins = np.histogram(frame[var].dropna(), density=True)
#             fig.add_histogram(
#                 x=frame[var],
#                 histnorm=histnorm,
#                 marker_color=color,
#                 marker_line_width=0,
#                 opacity=0.75,
#                 name=name,
#                 legendgroup=name,
#                 showlegend=k == 1,
#                 row=row,
#                 col=col,
#             )
#             fig.update_xaxes(title_text=var, row=row, col=col)

#     fig.update_xaxes(
#         tickfont_size=8,
#         showgrid=False,
#         titlefont_size=8,
#         titlefont_family="Arial Black",
#     )
#     fig.update_yaxes(tickfont_size=8, showgrid=False)

#     fig.update_layout(
#         width=840,
#         height=kwargs.get("height", 640),
#         title=kwargs.get("title", "Distributions"),
#         font_color=FONT_COLOR,
#         title_font_size=18,
#         plot_bgcolor=BACKGROUND_COLOR,
#         paper_bgcolor=BACKGROUND_COLOR,
#         bargap=kwargs.get("bargap", 0),
#         bargroupgap=kwargs.get("bargroupgap", 0),
#         legend=dict(
#             yanchor="bottom", xanchor="right", y=1, x=1, orientation="h", title=""
#         ),
#     )
#     return fig
