import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def discrete_cmap(N, vmin=0, vmax=1, base_cmap='seismic_r', over='w', under='w', bad='w', alpha=1.0, values=None):
    base = plt.cm.get_cmap(base_cmap)
    if values is None:
        color_list = np.array(base(np.linspace(vmin, vmax, N)))
    else:
        color_list = np.array(base(values))
    color_list[:,3] = alpha
    cmap_name = base.name + str(N)
    cmap = plt.cm.colors.ListedColormap(color_list, name=cmap_name)
    cmap.set_over(color=over, alpha=0)
    cmap.set_under(color=under, alpha=0)
    cmap.set_bad(color=bad, alpha=0)

    return cmap

def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    mins += deltas / 4
    maxs -= deltas / 4
    # mins += correction
    # maxs -= correction
    return mins, maxs, cs, deltas, tc, highs


def patch_3d_axis():
    from mpl_toolkits.mplot3d.axis3d import Axis
    if not hasattr(Axis, "_get_coord_info_old"):
        Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new


elev = 45
z_aspect = 2.5
fig_size = 8
z_locs = np.linspace(1, 7, 4).tolist()

mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams["axes.axisbelow"] = False
mpl.rcParams['mathtext.default'] = 'it'


def get_fixed_lim(mins, maxs, blend=0):
    og_mins = (49 * mins + maxs) / 50
    og_maxs = (mins + 49 * maxs) / 50
    
    mix_mins = blend * og_mins + (1 - blend) * mins
    mix_maxs = blend * og_maxs + (1 - blend) * maxs

    return mix_mins, mix_maxs

def style_3d_ax(ax):
    fake_axis_lines = []

    # Fix missing axis lines
    for z in z_locs:
        
        fake_axis_lines.append(ax.plot([0, 1], 
                                       [0, 0], 
                                       [z, z], **ax.xaxis._axinfo['axisline'], zorder=np.inf))
        fake_axis_lines.append(ax.plot([0, 1], 
                                       [1, 0], 
                                       [z, z], **ax.xaxis._axinfo['axisline'], zorder=np.inf))
        fake_axis_lines.append(ax.plot([0, 0], 
                                       [1, 0], 
                                       [z, z], **ax.xaxis._axinfo['axisline'], zorder=np.inf))
    
    zaxis_line_kwargs = dict(**ax.xaxis._axinfo['axisline'])
    zaxis_line_kwargs['color'] = (0, 0, 0, 0.2)
    
    for xy_locs in [
        ([0, 0], [0, 0]),
        ([0, 0], [1, 1]),
        ([1, 1], [0, 0]),
    ]:
        fake_axis_lines.append(ax.plot(*xy_locs, [z_locs[0], z_locs[-1]], 
                                       **zaxis_line_kwargs, zorder=np.inf if elev > 0 else None))
    ax.grid(False)
    
    ax.set_zticks(z_locs, minor=False)
    ax.set_zticklabels(reversed(c_nums))
    ax.view_init(elev, 180+45)
    ax.set_box_aspect((1, 1, z_aspect))
    
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], minor=False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_xticklabels([f'{t:g}' for t in ax.get_xticks()])
    ax.tick_params(axis='x', pad=0 if elev > 0 else 6)
    ax.tick_params(axis='y', pad=0 if elev > 0 else 6)
    ax.tick_params(axis='z', pad=8)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], minor=False)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_yticklabels([f'{t:g}' if t > 0 else '' for t in ax.get_yticks()])
    
    ax.set_proj_type('ortho')
    ax.set_zlim(*get_fixed_lim(min(z_locs), max(z_locs)))
    ax.set_xlim(*get_fixed_lim(0, 1))
    ax.set_ylim(*get_fixed_lim(0, 1))
    
    ax.w_zaxis.line.set_lw(0.)
    
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.925, 0.925, 0.925, 0.0))
        if axis.adir != 'z':
            axis.gridlines.set_segments([])
