# patched from https://github.com/shap/shap/blob/master/shap/plots/_beeswarm.py 
# to support multiple colormaps and specifying feature columns. 
    
# Summary plots of SHAP values across a whole dataset.

from __future__ import division

import warnings
import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from shap.plots._labels import labels
from shap.plots import colors
from shap.utils import safe_isinstance, OpChain, format_value
from shap.plots._utils import convert_ordering, convert_color, merge_nodes, get_sort_order, sort_inds
from shap import Explanation

def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text



def is_color_map(color):
    safe_isinstance(color, "matplotlib.colors.Colormap")


# TODO: remove unused title argument / use title argument
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def summary_legacy(shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
                 color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                 class_inds=None,
                 color_bar_label=labels["FEATURE_VALUE"], x_jitter=0.0, y_jitter=0.01,
                 cmap=colors.red_blue,
                 # depreciated
                 auto_size_plot=None,
                 use_log_scale=False, dot_size=16,
                  cmaps=None, plot_idx=None
                  ):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that 
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        base_value = shap_exp.base_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    # deprecation warnings
    if auto_size_plot is not None:
        warnings.warn("auto_size_plot=False is deprecated and is now ignored! Use plot_size=None instead.")

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if use_log_scale:
        pl.xscale('symlog')

    if max_display is None:
        max_display = 20
    
    if plot_idx is None:
        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
    else:
        feature_order = plot_idx

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="k", lw=1, zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            if cmaps is not None:
                color = cmaps[i]
                cmap = cmaps[i]
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]: # check categorical feature
                    colored_feature = False
                else:
                    values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * 1.333 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=dot_size, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                
                n_dots = len(shaps[np.invert(nan_mask)])
                x_jitter_arr = (np.random.rand(n_dots) - 0.5) * x_jitter
                y_jitter_arr = (np.random.rand(n_dots) - 0.5) * y_jitter
                pl.scatter(shaps[np.invert(nan_mask)] + x_jitter_arr, 
                           pos + ys[np.invert(nan_mask)] + ys[np.invert(nan_mask)] * y_jitter_arr,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=dot_size,
                           c=cvals, alpha=alpha, linewidth=0,
                           zorder=99999, rasterized=len(shaps) > 500)
            else:
                n_dots = len(shaps)
                x_jitter_arr = (np.random.rand(n_dots) - 0.5) * x_jitter
                y_jitter_arr = np.random.rand(n_dots) * y_jitter + 1.0
                pl.scatter(shaps + x_jitter_arr, pos + ys * y_jitter_arr, s=dot_size, alpha=alpha, linewidth=0, zorder=99999,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    else:
        raise ValueError('Unknown plot_type')

    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        import matplotlib.cm as cm
        
        
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, labelpad=0, fontsize=36)
        cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order])
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x',)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(labels['GLOBAL_VALUE'],)
    else:
        pl.xlabel(labels['VALUE'],)
    pl.tight_layout()
    if show:
        pl.show()
    