import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from pathlib import Path

# ---------------------------------
# Output path: script folder if possible
# ---------------------------------
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

output_path = script_dir / 'projector_population_rotation_side_by_side_clean.pdf'

# =============================
# Adjustable style parameters
# =============================
aps_full_width_in = 7.0
fig_width_in = 0.70 * aps_full_width_in
fig_height_in = 3.00

font_size = 10
unitary_color = '#0B6E4F'      # dark green
dissipator_color = '#8BC34A'   # light green
ghost_color = '0.65'
ghost_line_color = '0.75'
old_value_color = '0.55'

node_face_color = 'black'
node_edge_color = 'black'

label_box = dict(
    facecolor='white',
    edgecolor='none',
    pad=0.10,
    alpha=0.92
)

plt.rcParams.update({
    'font.size': font_size,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

# =============================
# Label switches
# =============================
show_all_labels = False

show_header_labels = show_all_labels
show_node_labels = show_all_labels
show_gamma_labels = True
show_mechanism_labels = False
show_stage_labels = True

# =============================
# Stage / arrow labels
# =============================
stage_1_label = r'$\rho_{\rm ss}[s_1]$'
stage_2_label = r'$\rho_{\rm ss}[s_2]$'

# =============================
# Example data
# =============================
p_t = np.array([0.40, 0.18, 0.14, 0.28])
p_tp = np.array([0.28, 0.12, 0.34, 0.26])

angles = np.deg2rad([90, 180, 270, 0])
R = 1.0
pos_t = np.c_[R * np.cos(angles), R * np.sin(angles)]

theta = np.deg2rad(22)
rot = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
pos_tp = pos_t @ rot.T

size_scale = 1350
sizes_t = size_scale * p_t
sizes_tp = size_scale * p_tp

# =============================
# Helper functions
# =============================
def unit_vector(v):
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([0.0, 0.0])
    return v / n

def tangent_perp(start, end):
    d = np.array(end) - np.array(start)
    t = unit_vector(d)
    p = np.array([-t[1], t[0]])
    return t, p

def point_on_circle(r, deg):
    th = np.deg2rad(deg)
    return np.array([r * np.cos(th), r * np.sin(th)])

def external_label_position(x, y, r_offset=0.40, t_offset=0.0):
    vec = np.array([x, y], dtype=float)
    vec = vec / np.linalg.norm(vec)
    tan = np.array([-vec[1], vec[0]])
    pos = np.array([x, y]) + r_offset * vec + t_offset * tan

    if vec[0] > 0.30:
        ha = 'left'
    elif vec[0] < -0.30:
        ha = 'right'
    else:
        ha = 'center'

    if vec[1] > 0.30:
        va = 'bottom'
    elif vec[1] < -0.30:
        va = 'top'
    else:
        va = 'center'

    return pos[0], pos[1], ha, va

def add_text(ax, x, y, s, color='black', fs=None, ha='center', va='center'):
    if fs is None:
        fs = font_size
    ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va, bbox=label_box)

def draw_arrow(ax, start, end, color, rad=0.0, lw=1.4, ms=8,
               shrinkA=0, shrinkB=0, alpha=1.0, zorder=1):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle='->',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f'arc3,rad={rad}',
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)

def marker_radius_pixels(s, dpi):
    """
    Approximate scatter-circle radius in pixels from area s in pt^2.
    """
    r_pt = np.sqrt(s / np.pi)
    return r_pt * dpi / 72.0

def draw_trimmed_straight_arrow(ax, start, end, size_start, size_end,
                                color, offset_px=0.0,
                                lw=1.5, ms=8.0, pad_px=4.0, zorder=1):
    """
    Draw a straight arrow offset slightly and trimmed so it does not
    intersect the marker discs.
    Returns display-coordinate endpoints and unit/perp vectors for labels.
    """
    start_disp = ax.transData.transform(start)
    end_disp = ax.transData.transform(end)

    d = end_disp - start_disp
    dist = np.linalg.norm(d)
    if dist == 0:
        return None

    t = d / dist
    p = np.array([-t[1], t[0]])

    r0 = marker_radius_pixels(size_start, ax.figure.dpi) + pad_px
    r1 = marker_radius_pixels(size_end, ax.figure.dpi) + pad_px

    start2_disp = start_disp + offset_px * p + r0 * t
    end2_disp   = end_disp   + offset_px * p - r1 * t

    if np.linalg.norm(end2_disp - start2_disp) < 4:
        return None

    inv = ax.transData.inverted()
    start2 = inv.transform(start2_disp)
    end2   = inv.transform(end2_disp)

    draw_arrow(
        ax, start2, end2,
        color=color, rad=0.0, lw=lw, ms=ms,
        shrinkA=0, shrinkB=0, zorder=zorder
    )

    return start2_disp, end2_disp, t, p

def add_gamma_label_display(ax, start_disp, end_disp, text,
                            perp_shift_px=12, along_shift_px=0,
                            color='black', fs=9):
    if not show_gamma_labels:
        return

    t = unit_vector(end_disp - start_disp)
    p = np.array([-t[1], t[0]])
    mid = 0.5 * (start_disp + end_disp)
    pos_disp = mid + along_shift_px * t + perp_shift_px * p
    pos_data = ax.transData.inverted().transform(pos_disp)

    add_text(ax, pos_data[0], pos_data[1], text, color=color, fs=fs)

def add_mixed_transition_label(ax, x, y, n, p_old, p_new):

    # Put every label to the left of its disc
    lx = x+0.3
    ly = y-0.28
    ha = 'right'
    va = 'center'

    pieces = [
        TextArea(rf'$p_{n}:$',
                 textprops=dict(color='black',
                                fontsize=font_size - 1)),

        TextArea(f'{p_old:.2f}',
                 textprops=dict(color=old_value_color,
                                fontsize=font_size - 1)),

        TextArea(r'$\to$',
                 textprops=dict(color='black',
                                fontsize=font_size - 1)),

        TextArea(f'{p_new:.2f}',
                 textprops=dict(color='black',
                                fontsize=font_size - 1)),
    ]

    packed = HPacker(
        children=pieces,
        align='center',
        pad=0,
        sep=2
    )

    ab = AnnotationBbox(
        packed,
        (lx, ly),
        xycoords='data',
        frameon=True,
        box_alignment=(1.0, 0.5),   # right edge of label anchored at lx
        bboxprops=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.92
        )
    )

    ax.add_artist(ab)

def draw_basis_motion(ax, old_positions, new_positions, old_sizes):
    ax.scatter(
        old_positions[:, 0], old_positions[:, 1],
        s=old_sizes,
        facecolors='none',
        edgecolors=ghost_color,
        linewidths=1.0,
        alpha=0.80,
        zorder=0
    )

    for old, new in zip(old_positions, new_positions):
        ax.plot(
            [old[0], new[0]],
            [old[1], new[1]],
            linestyle='--',
            linewidth=1.0,
            color=ghost_line_color,
            alpha=0.95,
            zorder=0
        )

def draw_population_arrows(ax, positions, sizes):
    """
    Straight dissipative arrows trimmed to avoid the discs.
    Gamma_13 and Gamma_31 are separated.
    """
    # Simple arrows
    simple_arrows = [
        # i, j, label, perp_shift_px, along_shift_px, offset_px
        (1, 2, r'$\Gamma_{12}$',  12,  -20,   0),
        (0, 2, r'$\Gamma_{02}$', 0,   -70,   0),
        (2, 3, r'$\Gamma_{23}$',  12,   -20,   0),
    ]

    for i, j, label, perp_shift_px, along_shift_px, offset_px in simple_arrows:
        result = draw_trimmed_straight_arrow(
            ax,
            positions[i], positions[j],
            sizes[i], sizes[j],
            color=dissipator_color,
            offset_px=offset_px,
            lw=1.5, ms=8.0, pad_px=4.0, zorder=1
        )
        if result is not None:
            start_disp, end_disp, _, _ = result
            add_gamma_label_display(
                ax, start_disp, end_disp, label,
                perp_shift_px=perp_shift_px,
                along_shift_px=along_shift_px,
                color=dissipator_color,
                fs=font_size - 1
            )

    # Separated pair: 1 -> 3 and 3 -> 1
    result = draw_trimmed_straight_arrow(
        ax,
        positions[1], positions[3],
        sizes[1], sizes[3],
        color=dissipator_color,
        offset_px=+14,
        lw=1.5, ms=8.0, pad_px=4.0, zorder=1
    )
    if result is not None:
        start_disp, end_disp, _, _ = result
        add_gamma_label_display(
            ax, start_disp, end_disp, r'$\Gamma_{13}$',
            perp_shift_px=+10,
            along_shift_px=-40,
            color=dissipator_color,
            fs=font_size - 1
        )

    result = draw_trimmed_straight_arrow(
        ax,
        positions[3], positions[1],
        sizes[3], sizes[1],
        color=dissipator_color,
        offset_px=+14,
        lw=1.5, ms=8.0, pad_px=4.0, zorder=1
    )
    if result is not None:
        start_disp, end_disp, _, _ = result
        add_gamma_label_display(
            ax, start_disp, end_disp, r'$\Gamma_{31}$',
            perp_shift_px=+10,
            along_shift_px=-40,
            color=dissipator_color,
            fs=font_size - 1
        )

def draw_unitary_rotation(ax):
    """
    Two counterclockwise arrows with matching curvature.
    """
    r_arc = 1.5
    rad = 0.34

    # start1 = point_on_circle(r_arc, 20)
    # end1   = point_on_circle(r_arc, 120)
    # draw_arrow(
    #     ax, start1, end1,
    #     color=unitary_color,
    #     rad=rad,
    #     lw=1.7,
    #     ms=9,
    #     zorder=1
    # )

    start2 = point_on_circle(r_arc, -70)
    end2   = point_on_circle(r_arc, 30)
    draw_arrow(
        ax, start2, end2,
        color=unitary_color,
        rad=rad,
        lw=1.7,
        ms=9,
        zorder=1
    )

def draw_nodes_and_labels(ax, positions, populations, sizes,
                          old_populations=None,
                          show_transition_values=False):
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=sizes,
        facecolor=node_face_color,
        edgecolor=node_edge_color,
        linewidth=0.7,
        zorder=3
    )

    if show_node_labels:
        for n, ((x, y), p) in enumerate(zip(positions, populations)):
            lx, ly, ha, va = external_label_position(x, y, r_offset=0.42) #0.42
            text = rf'$\Pi_{n}$' + '\n' + rf'$p_{n}={p:.2f}$'
            add_text(ax, lx, ly, text, fs=font_size, ha=ha, va=va)

    if show_transition_values and old_populations is not None:
        for n, ((x, y), p_old, p_new) in enumerate(zip(positions, old_populations, populations)):
            add_mixed_transition_label(ax, x, y, n, p_old, p_new)

def draw_panel(ax, positions, populations, sizes, title,
               ghost_positions=None, ghost_sizes=None,
               show_green_processes=True,
               show_transition_values=False,
               old_populations=None):
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.38, 1.38)

    if show_stage_labels:
        ax.text(0.5, 1.02, title, transform=ax.transAxes,
                ha='center', va='bottom')

    if ghost_positions is not None and ghost_sizes is not None:
        draw_basis_motion(ax, ghost_positions, positions, ghost_sizes)

    if show_green_processes:
        draw_population_arrows(ax, positions, sizes)
        draw_unitary_rotation(ax)

    draw_nodes_and_labels(
        ax, positions, populations, sizes,
        old_populations=old_populations,
        show_transition_values=show_transition_values
    )

def add_middle_label(overlay_ax):
    """
    Two-line label above the middle arrow:
    Evolve under
    A = A^U + A^D
    with U and D terms colored.
    """
    line1 = TextArea(
        'Evolve under',
        textprops=dict(color='black', fontsize=font_size)
    )

    line2_pieces = [
        TextArea(r'$\mathcal{A}$', textprops=dict(color='black', fontsize=font_size)),
        TextArea(r'$=$',           textprops=dict(color='black', fontsize=font_size)),
        TextArea(r'$\mathcal{A}^{\rm U}$', textprops=dict(color=unitary_color, fontsize=font_size)),
        TextArea(r'$+$',           textprops=dict(color='black', fontsize=font_size)),
        TextArea(r'$\mathcal{A}^{\rm D}$', textprops=dict(color=dissipator_color, fontsize=font_size)),
    ]
    line2 = HPacker(children=line2_pieces, align='center', pad=0, sep=2)

    label_box_obj = VPacker(children=[line1, line2], align='center', pad=0, sep=2)

    ab = AnnotationBbox(
        label_box_obj,
        (0.50, 0.575),
        xycoords='data',
        frameon=False,
        box_alignment=(0.5, 0.0)
    )
    overlay_ax.add_artist(ab)

# =============================
# Build figure
# =============================
fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))

# Give the panels more horizontal breathing room before doing display-coordinate trimming
plt.subplots_adjust(left=-0.00, right=0.98, top=0.82, bottom=0.12, wspace=0.60)

if show_header_labels:
    fig.text(
        0.50, 0.98,
        r'$\rho_{\rm ss}(s)=\sum_n p_n(s)\,\Pi_n(s)$',
        ha='center', va='top',
        fontsize=font_size + 1
    )

# Left panel
draw_panel(
    axes[0],
    pos_t, p_t, sizes_t,
    stage_1_label,
    show_green_processes=True,
    show_transition_values=False
)

# Right panel
draw_panel(
    axes[1],
    pos_tp, p_tp, sizes_tp,
    stage_2_label,
    ghost_positions=pos_t,
    ghost_sizes=sizes_t,
    show_green_processes=False,
    show_transition_values=True,
    old_populations=p_t
)

# Middle arrow / overlay
overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
overlay.set_axis_off()
overlay.set_xlim(0, 1)
overlay.set_ylim(0, 1)

overlay.annotate(
    '',
    xy=(0.64, 0.47), xytext=(0.36, 0.47),
    xycoords='data', textcoords='data',
    arrowprops=dict(arrowstyle='->', lw=1.8, color='black')
)

add_middle_label(overlay)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved to: {output_path}")