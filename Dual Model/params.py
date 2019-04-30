class Params:
    n_tokens = 9
    n_items = 26
    n_dual_train_items = 9
    max_list_len = 9
    test_list_len = 6
    sr_stop_crit = 0.5
    bp_stop_crit = 0.999
    tests = [0, 1, 2, 3]
    
assert(Params.n_tokens >= Params.max_list_len)    