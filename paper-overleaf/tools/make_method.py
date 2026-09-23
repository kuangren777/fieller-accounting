#!/usr/bin/env python3
"""Draw the method's data flow; no experimental data or simulation required.

Run: python paper-overleaf/tools/make_method.py
Outputs: figs/fig_method.{pdf,svg,png} at column width, plus fig_method_wide.*.
Solid arrows carry observations/sets; the dashed arrow feeds an index back.
The sample-size rule is a separate design option, not an online policy step.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
INK, BLUE, GRAY = '#262b30', '#315e78', '#66717a'


def _text(ax, x, y, value, size=7.5, **kw):
    return ax.text(x, y, value, fontsize=size, va='center', color=INK, **kw)


def _module(ax, x, y, w, h, title, accent=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white',
                           edgecolor=BLUE if accent else INK, linewidth=.65))
    ax.add_patch(Rectangle((x, y+h-.28), w, .28, facecolor='#edf2f5' if accent else '#f2f3f3',
                           edgecolor='none'))
    ax.plot([x, x+w], [y+h-.28]*2, color=BLUE if accent else INK, lw=.45)
    _text(ax, x+.10, y+h-.14, title, size=8, weight='bold')


def _arrow(ax, points, dashed=False):
    # Orthogonal segments with a single small arrowhead at the destination.
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=BLUE if dashed else INK, lw=.75,
            ls=(0, (4, 2.5)) if dashed else '-', solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=6, linewidth=.75,
                                linestyle=(0, (4, 2.5)) if dashed else '-',
                                color=BLUE if dashed else INK, shrinkA=0, shrinkB=0))


def _column_figure():
    fig = plt.figure(figsize=(3.45, 1.94))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(-.13, 3.52), ylim=(-.10, 3.06))
    ax.axis('off')
    # Sampler and observations form the top of the feedback loop.
    ax.add_patch(Rectangle((.25, 2.38), 3.14, .56, facecolor='#f2f3f3',
                           edgecolor=INK, linewidth=.65))
    _text(ax, .36, 2.81, 'Budgeted sampler', size=8, weight='bold')
    _text(ax, .36, 2.56, r'$a_t=\arg\max_a U_a$', size=7)
    _text(ax, 2.21, 2.64, r'$\sum_t c_t < B$', size=7)
    _arrow(ax, [(1.0, 2.38), (1.0, 2.08)])
    _text(ax, 1.16, 2.27, r'$n,\bar r,\bar c,s_{rr},s_{cc},s_{rc}$', size=7)

    _module(ax, .25, 1.17, 1.42, .91, 'M1  Existence')
    _text(ax, .96, 1.61, r'$A=\bar c^2-t^2s_{cc}/n$', size=6.8, ha='center')
    _text(ax, .96, 1.39, r'$A>0\quad\wedge\quad\bar r>0$', size=7, ha='center')
    _module(ax, 1.96, 1.17, 1.43, .91, 'M2  Fieller set', accent=True)
    _text(ax, 2.675, 1.64, r'$A\theta^2+B\theta+C\leq0$', size=7, ha='center')
    # Distinct glyphs encode set topology, not a numerical result.
    ax.plot([2.08, 2.42], [1.45, 1.45], color=BLUE, lw=1.3)
    for x in (2.08, 2.42):
        ax.plot([x, x], [1.425, 1.475], color=BLUE, lw=.65)
    _text(ax, 2.52, 1.45, r'$[\ell,u]$  or  $\{0\}$', size=6.8)
    _arrow(ax, [(2.20, 1.27), (2.07, 1.27)])
    _arrow(ax, [(2.31, 1.27), (2.44, 1.27)])
    _text(ax, 2.52, 1.27, 'unbounded', size=6.8)
    _arrow(ax, [(1.67, 1.60), (1.96, 1.60)])

    # A shared set output branches to accounting and to the fallback.
    _arrow(ax, [(2.675, 1.17), (2.675, .87)])
    ax.plot([.96, 2.675], [1.02, 1.02], color=INK, lw=.75)
    ax.plot([2.675], [1.02], 'o', color=INK, ms=1.7)
    _arrow(ax, [(.96, 1.02), (.96, .87)])
    _text(ax, 2.82, 1.03, r'$I_a$', size=7)
    _module(ax, .25, .16, 1.42, .71, 'M4  Index fallback', accent=True)
    _text(ax, .34, .44, r'$U_a=u_a$  if $q_{\rm nd}=1$', size=6.7)
    _text(ax, .34, .29, r'$U_a=u_a^{\mathrm{delta\! -\! JV}}$  otherwise', size=6.2)
    _module(ax, 1.96, .16, 1.43, .71, 'M3  Accounting')
    _text(ax, 2.06, .44, r'$q_{\rm nd}$: positive-width interval', size=6.1)
    _text(ax, 2.06, .29, r'$q_{\rm b}$: nonempty bounded set', size=6.1)
    _arrow(ax, [(.25, .43), (.07, .43), (.07, 2.70), (.25, 2.70)], dashed=True)
    _text(ax, .14, 1.02, r'$U_a$', size=6.3, rotation=90, ha='center',
          bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .3})
    _arrow(ax, [(2.675, .16), (2.675, .055)])
    _text(ax, 2.56, .052, 'report / coverage', size=6, ha='right')
    return fig


def main():
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(7.1, 2.18))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(-.10, 7.18), ylim=(-.12, 2.46))
        ax.axis('off')

        # Collection and sufficient statistics.
        _module(ax, .06, 1.10, 1.24, 1.22, 'Budgeted sampler')
        _text(ax, .68, 1.86, r'$a_t=\arg\max_a\ U_a$', ha='center')
        _text(ax, .68, 1.61, r'$r_t\in\{0,1\},\ c_t>0$', size=7.2, ha='center')
        ax.plot([.16, 1.20], [1.43]*2, lw=.4, color=GRAY)
        _text(ax, .68, 1.27, r'$\sum_t c_t < B$', ha='center')
        _text(ax, .68, .91, 'Per-arm moments', size=7, ha='center')
        _text(ax, .68, .72, r'$n,\bar r,\bar c,s_{rr},s_{cc},s_{rc}$', size=6.8, ha='center')
        _arrow(ax, [(1.30, 1.68), (1.59, 1.68)])

        _module(ax, 1.59, 1.10, 1.49, 1.22, 'M1  Existence screen')
        _text(ax, 2.335, 1.84, r'$A=\bar c^2-t^2s_{cc}/n$', ha='center')
        _text(ax, 2.335, 1.57, r'$A>0\quad\wedge\quad\bar r>0$', ha='center')
        _text(ax, 2.335, 1.28, 'Positive-width interval', size=7, ha='center')
        _arrow(ax, [(3.08, 1.68), (3.37, 1.68)])

        _module(ax, 3.37, 1.10, 1.66, 1.22, 'M2  Fieller set', accent=True)
        _text(ax, 4.20, 1.86, r'$I_a=\{\theta:A\theta^2+B\theta+C\leq0\}$', size=6.7, ha='center')
        # Set glyphs are schematic topologies, not measured intervals.
        for y, kind, label in [(1.62, 'interval', r'$[\ell,u]$'),
                               (1.42, 'point', r'$\{0\}$'),
                               (1.22, 'unbounded', 'unbounded')]:
            lo, hi = 3.51, 4.12
            if kind == 'interval':
                ax.plot([lo, hi], [y, y], color=BLUE, lw=1.5)
                ax.plot([lo, lo], [y-.035, y+.035], color=BLUE, lw=.75)
                ax.plot([hi, hi], [y-.035, y+.035], color=BLUE, lw=.75)
            elif kind == 'point':
                ax.plot([lo, hi], [y, y], color='#bdc4c9', lw=.45)
                ax.plot([(lo+hi)/2], [y], 'o', color=BLUE, ms=2.7)
            else:
                _arrow(ax, [(lo+.19, y), (lo, y)])
                _arrow(ax, [(hi-.19, y), (hi, y)])
            _text(ax, 4.27, y, label, size=7)
        _arrow(ax, [(5.03, 1.68), (5.35, 1.68)])
        _text(ax, 5.19, 1.83, r'$I_a$', size=7, ha='center')

        _module(ax, 5.35, 1.10, 1.69, 1.22, 'M3  Report accounting')
        _text(ax, 5.46, 1.85, 'Report mask', size=7, style='italic')
        _text(ax, 5.46, 1.60, r'$q_{\rm nd}=\mathbf{1}\{\ell<u,\; I_a\ \mathrm{bounded}\}$', size=6.6)
        _text(ax, 5.46, 1.30, r'$q_{\rm b}=\mathbf{1}\{I_a\ne\varnothing,\; I_a\ \mathrm{bounded}\}$', size=6.6)
        _arrow(ax, [(6.195, 1.10), (6.195, .79)])
        _text(ax, 6.195, .61, 'Report rate / coverage', size=7.5, ha='center')
        _text(ax, 6.195, .38, 'conditional · joint · set', size=7, ha='center')

        # Index construction receives the set directly, not accounting results.
        _module(ax, 1.59, .16, 3.44, .64, 'M4  Index fallback', accent=True)
        _text(ax, 1.70, .34, r'$U_a=u_a\ \mathrm{if}\ q_{\rm nd}=1;\quad U_a=u_a^{\mathrm{delta\! -\! JV}}\ \mathrm{otherwise}$', size=7.3)
        _arrow(ax, [(4.20, 1.10), (4.20, .80)])
        _text(ax, 4.33, .95, r'$I_a$', size=7)
        _arrow(ax, [(1.59, .47), (.10, .47), (.10, 1.10)], dashed=True)
        _text(ax, .86, .31, r'index $U_a$', size=7, ha='center')
        _text(ax, 2.72, .95, r'$n_0=\lceil\log\eta/\log(1-\tilde\mu)\rceil$', size=6.8, ha='center')
        # Optional allocation design; no edge into the online feedback loop.
        ax.text(1.67, .95, 'design:', fontsize=6.8, va='center', color=GRAY)

        dest = ROOT / 'figs'
        dest.mkdir(exist_ok=True)
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(dest / f'fig_method_wide.{ext}', dpi=300, facecolor='white',
                        bbox_inches='tight', pad_inches=.02)
        plt.close(fig)
        fig = _column_figure()
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(dest / f'fig_method.{ext}', dpi=300, facecolor='white',
                        bbox_inches='tight', pad_inches=.02)
        plt.close(fig)
        print('Wrote column and wide method figures (PDF, SVG, PNG)')


if __name__ == '__main__':
    main()
