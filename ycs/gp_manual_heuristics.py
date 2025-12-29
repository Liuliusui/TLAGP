from deap import gp

try:
    from ycs.my_gp import primitive_set  # absolute package import
except ImportError:
    from .my_gp import primitive_set  # fallback for relative execution

k1_list = [0.2, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 5.6, 6, 6.4, 6.8, 7.2]
k2_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
k3_list = [0.001, 0.0025, 0.004, 0.005, 0.025, 0.04, 0.05, 0.25, 0.4, 0.6, 0.8, 1, 1.2]


def atcrcs(k1, k2, k3):
    weighted_shortest_processing_time_term = 'div (w_i, add (add (p_i, s_ji), max_0 (sub (r_i, t))))'
    # slack_term = '1'
    slack_term = 'exp (neg (div (max_0 (sub (sub (d_i, p_i), t)), mul (' + str(k1) + ', add (bar_p, bar_s)))))'
    setup_term = 'exp (neg (div (s_ji, mul (' + str(k2) + ', bar_s))))'
    ready_term = 'exp (neg (div (max_0 (sub (r_i, t)), mul (' + str(k3) + ', bar_p))))'

    atcrcs_str = ('mul (mul (mul (' + weighted_shortest_processing_time_term + ', ' +
                  slack_term + '), ' + setup_term + '), ' + ready_term + ')')
    return atcrcs_str


def atcrss(k1, k2, k3):
    weighted_shortest_processing_time_term = 'div (w_i, add (p_i, max (s_ji, sub (r_i, t))))'
    # slack_term = '1'
    slack_term = 'exp (neg (div (max_0 (sub (sub (d_i, p_i), t)), mul (' + str(k1) + ', add (bar_p, bar_s)))))'
    setup_term = 'exp (neg (div (s_ji, mul (' + str(k2) + ', bar_s))))'
    ready_term = 'exp (neg (div (max_0 (sub (r_i, t)), mul (' + str(k3) + ', bar_p))))'

    atcrss_str = ('mul (mul (mul (' + weighted_shortest_processing_time_term + ', ' +
                  slack_term + '), ' + setup_term + '), ' + ready_term + ')')
    return atcrss_str


def atcsr(k1, k2, k3):
    weighted_shortest_processing_time_term = 'div (w_i, p_i)'
    # "exp (neg ((max_zero ((d_i - p_i) - (r_i max t))) / (" + k1_s + " * bar_p)))"
    slack_term = 'exp (neg (div (max_0 (sub (sub (d_i, p_i), max (r_i, t))), mul (' + str(k1) + ', bar_p))))'
    setup_term = 'exp (neg (div (s_ji, mul (' + str(k2) + ', bar_s))))'
    ready_term = 'exp (neg (div (max_0 (sub (r_i, t)), mul (' + str(k3) + ', bar_p))))'

    atcrs_str = ('mul (mul (mul (' + weighted_shortest_processing_time_term + ', ' +
                 slack_term + '), ' + setup_term + '), ' + ready_term + ')')
    return atcrs_str

def get_all_atc_rules():
    atc_rules = []
    for k1 in k1_list:
        for k2 in k2_list:
            for k3 in k3_list:
                atcsr_rule = gp.compile(atcsr(k1, k2, k3), primitive_set)
                atcrss_rule = gp.compile(atcrss(k1, k2, k3), primitive_set)
                atcrcs_rule = gp.compile(atcrcs(k1, k2, k3), primitive_set)

                atc_rules.append(atcsr_rule)
                atc_rules.append(atcrss_rule)
                atc_rules.append(atcrcs_rule)
    return atc_rules


fcfs_rule = gp.compile('div (1.0, r_i)', primitive_set)
ntf_rule = gp.compile('div (1.0, s_ji)', primitive_set)
edd_rule = gp.compile('div (1.0, d_i)', primitive_set)
sctf_rule = gp.compile('div (1.0, add (r_i, p_i))', primitive_set)
mwspt_rule = gp.compile('div (w_i, add (p_i, s_ji))', primitive_set)

atcsr_rule = gp.compile(atcsr(2.8, 1.9, 0.4), primitive_set)
atcrss_rule = gp.compile(atcrss(2.0, 1.9, 0.6), primitive_set)
atcrcs_rule = gp.compile(atcrcs(1.8, 2.1, 0.8), primitive_set)

gp_evolved = gp.compile('sub(sub(neg(add(add(1.0, max(add(t, s_ji), sub(d_i, p_i))), 1.0)), add(add(max(max(add(t, s_ji), r_i), t), div(max(1.0, max(t, t)), mul(div(w_i, p_i), t))), div(t, mul(div(w_i, div(p_i, w_i)), add(t, s_ji))))), add(add(neg(w_i), add(add(div(p_i, bar_p), add(div(p_i, bar_p), bar_p)), max(r_i, add(max_0(t), s_ji)))), add(add(1.0, add(max(t, r_i), max(add(t, s_ji), r_i))), add(t, max(r_i, add(t, s_ji))))))', primitive_set)

best_manual_rules = [fcfs_rule, ntf_rule, edd_rule, sctf_rule, mwspt_rule, atcsr_rule, atcrss_rule, atcrcs_rule]
all_manual_rules = best_manual_rules + get_all_atc_rules()
best_manual_rules = [('FCFS', fcfs_rule), ('NTF', ntf_rule), ('EDD', edd_rule), ('SCTF', sctf_rule),
                     ('MWSPT', mwspt_rule), ('ATCSR', atcsr_rule), ('ATCRSS', atcrss_rule), ('ATCRCS', atcrcs_rule),
                     ('GP', gp_evolved)]
simple_manual_rules = [('FCFS', fcfs_rule), ('NTF', ntf_rule), ('EDD', edd_rule), ('SCTF', sctf_rule),
                     ('MWSPT', mwspt_rule)]


if __name__ == "__main__":
    print(atcsr(1.0, 1.0, 1.0))
    print(atcsr_rule(**{'r_i': 202.24, 'p_i': 2, 's_ji': 0, 'd_i': 204.24, 'w_i': 1, 't': 202.24, 'n': 1, 'bar_r': 202.24, 'bar_p': 2.0, 'bar_s': 0.0, 'bar_d': 204.24, 'bar_w': 1.0}))
