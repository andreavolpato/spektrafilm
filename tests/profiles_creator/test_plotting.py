from spektrafilm_profile_creator.plotting import plot_profile
import spektrafilm_profile_creator.plotting as plotting_module

from tests.profiles_creator.helpers import make_test_profile


def test_plot_profile_draws_into_supplied_axes() -> None:
    profile = make_test_profile('stock_a')
    figure, axes = plotting_module.plt.subplots(1, 3)

    returned_figure, returned_axes = plot_profile(profile, figure=figure, axes=axes)

    assert returned_figure is figure
    assert len(returned_axes) == 3
    assert returned_axes[0].figure is figure
    assert figure.texts[0].get_text() == 'stock_a - stock_a'
