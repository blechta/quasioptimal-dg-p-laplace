import sys
import os
from ast import literal_eval


discretizations = {
    'DG': ('DG', r'the primal IIDG formulation, i.e., \eqref{eq:discrete_primal_DG} with $\tilde{\nabla}_{h_{\ell}}=\nabla_{h_{\ell}}$'),
    'LDG': ('LDG', r'the primal LDG formulation, i.e., \eqref{eq:discrete_primal_DG} with $\tilde{\nabla}_{h_{\ell}}=\fG_{h_{\ell}}^1$'),
    'CR': ('CR', r'the primal Crouzeix--Raviart formulation (without jump stabilisation), i.e., $\tilde{\nabla}_{h_{\ell}}=\nabla_{h_{\ell}}$ and $\alpha=0$'),
    'mDG': ('mixed_DG', r'the mixed LDG formulation'),
}

rhos = {
    '1.0': ('one',   r'$\beta=1.01               $ and, thus, $\rho=1.0$'),
    '0.5': ('half',  r'$\beta=1.01 - \frac{1}{p} $ and, thus, $\rho=0.5$'),
    '0.2': ('fifth', r'$\beta=1.01 - \frac{8}{5p}$ and, thus, $\rho=0.2$'),
}

ps = ('1.5', '1.7', '2.0', '3.0', '4.5')

# NB: mesh sizes hardcoded here: needs change if changing initial mesh
hs = ('0.1668', '0.0834', '0.0417', '0.0208', '0.0104', '0.0052', '0.0026')

table_snippet = r"""
\newcommand*%s{%%
  \begin{table}[H]
    \setlength\tabcolsep{8pt}
    \centering
    \begin{tabular}{%s}
      \hline
      \cellcolor{lightgray}\diagbox[height=1.1\line,width=0.1275\dimexpr\linewidth]{\vspace{-0.5mm}\hspace*{-2mm}$h_\ell$}{\\[-5mm] $p$\hspace*{-2mm}}
      %s
      \\ \hline\hline
      %s
      \hline
      \cellcolor{lightgray}\small\textrm{Expected}
      %s
      \\ \hline
    \end{tabular}
    \vspace{1ex}
    \caption{Experimental order of convergence EOC$_\ell$, $\ell \in \{1,\ldots,%s\}$,
             for %s,
             with %s.}
    \label{%s}
  \end{table}
}
"""


def generate_table_code(disc, rho, rates):
    col_spec = '|c||' + '|'.join('c' if p != '2.0' else '|c|' for p in ps) + '|'
    ps_code = ' '.join(r'& \cellcolor{lightgray}' + p for p in ps)
    rates_code = '\n      '.join(
        r'\cellcolor{lightgray}' + f'${h}$ ' + ' '.join(f'& ${rates[p][i]:.3f}$' for p in ps) + r' \\ \hline'
        for i, h in enumerate(hs[1:])
    )
    expected_rates_code = ' '.join(r'& ' + rho for p in ps)
    num_refs = max(len(rates[p]) for p in ps)
    assert num_refs == len(hs[1:])
    disc_tag, disc_desc = discretizations[disc]
    rho_tag, rho_desc = rhos[rho]
    tag = f'tbl:{disc_tag}_rate_{rho}'
    cmd_name = fr'\tbl_{disc_tag}_rate_{rho_tag}'.title().replace('_', '')
    code = table_snippet % (cmd_name, col_spec, ps_code, rates_code,
                            expected_rates_code, num_refs, disc_desc,
                            rho_desc, tag)
    return code



def main(outdir):

    rates = extract_rates(outdir)

    code = []
    for disc, rates_disc in rates.items():
        for rho, rates_disc_rho in rates_disc.items():
            code_table = generate_table_code(disc, rho, rates_disc_rho)
            code_table = code_table.strip()
            code.append(code_table)

    code = '\n\n'.join(code)

    print(code)


def extract_rates(outdir):
    rates = {}
    for disc in discretizations:
        rates[disc] = {}
        for rho in rhos:
            rates[disc][rho] = {}
            for p in ps:
                filepath = os.path.join(outdir, f'{disc}_rate_{rho}_p_{p}.log')
                r = parse_log_file_for_rates(filepath)
                r = r['total']
                rates[disc][rho][p] = r
    return rates


def parse_log_file_for_rates(filepath):

    with open(filepath, 'rt') as f:

        # Seek computed rates
        for line in f:
            if line.startswith('Computed rates:'):
                break

        # Reed all rates
        lines = []
        for line in f:
            if line.startswith('Average EOC:'):
                # End of section with rates
                break
            if len(line.strip()) == 0:
                continue
            lines.append(line)

    assert len(lines) > 0, f"Computed rates not found in {filepath}"
    assert lines[0].lstrip()[0] == '{', f"Rates extracted from {filepath} malformed"
    assert lines[-1].rstrip()[-1] == '}', f"Rates extracted from {filepath} malformed"

    code = ''.join(lines)
    code = code.replace('nan', 'None')
    rates = literal_eval(code)

    return rates


if __name__ == '__main__':
    try:
        _, outdir = sys.argv
    except ValueError:
        raise ValueError('Expecting a directory path as a sole command-line argument')
    main(outdir)
