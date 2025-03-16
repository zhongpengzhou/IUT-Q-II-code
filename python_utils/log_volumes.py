from .basic import *

def log_volume_at_p(l, p, ep, e0, reduction_type):
    # compute the upper bound of "certain contribution of p in Vol(l)" for given ep
    l1 = l // 2
    upper_bound = 0
    if reduction_type == 'good':
        is_multi = 0
    elif reduction_type == 'multi':
        is_multi = 1
    else:
        raise ValueError('reduction type invalid')
    # p-adic valuation of ep
    vp_ep = compute_valuation(ep, p)
    # dp
    if vp_ep == 0:
        dp = 1 - Fraction(1, ep)
    else:
        dp = vp_ep + 1
    if ep==2 and p==2:
        dp = 1
    # ap = (ep // (p-1) + 1) / ep, bp = max{n - p^n / ep}
    ap = Fraction(ep//(p-1) + 1, ep)
    if ep <= p-1:
        bp = Fraction(-1,ep)
    elif ep <= p*(p-1):
        bp = 1-Fraction(p,ep)
    else:
        n0 = ceil(log(Fraction(ep,p-1))/log(p))
        bp = n0 - Fraction(pow(p,n0),ep)

    # the contribution at vp in \log\rad(N)
    if ep % l != 0:
        cp = is_multi * (1 - Fraction(1, e0))         
    else: # if ep % l == 0:
        cp = is_multi * (1 - Fraction(1, e0*l))
        

    # the upper bound of B_1(p)
    if is_multi == 0 or ep % l != 0:
        # In this case, v_p(N) \equiv 0 mod l
        for j in range(1,l1+1):
            upper_bound += max(ceil(j*dp+(j+1)*ap), (j+1)*(1+(p==2)))+ (j+1)*bp
        upper_bound = Fraction(upper_bound, l1) / Fraction(l+5, 4) - cp
        return upper_bound
    if ep % l == 0:
        # the largest B_1(p) for vp(N) mod l in [1,2,...,l-1].
        tmp_upper_bounds = []
        for vp in range(1, 2* l):
            tmp_upper_bound = 0
            for j in range(1,l1+1):
                lambda1 = Fraction(-j*j*vp, 2* l)
                tmp_upper_bound += max(ceil(lambda1+j*dp+(j+1)*ap) - lambda1, (j+1)*(1+(p==2))) + (j+1)*bp
            tmp_upper_bounds.append(tmp_upper_bound)
        upper_bound = max(tmp_upper_bounds) / Fraction(l+5, 4)
        upper_bound = Fraction(upper_bound, l1) - cp
        return upper_bound

# example of ramification datasets
def get_ramification_dataset_example(l):
    # the case of semi-stable Frey-Hellegouarch curves
    ep_range = dict()
    ep_range['general'] = {'good':[1], 'multi':[1,3,l,3*l]} # general ramification indices, e0 = 3
    ep_range[2] = {'good':[2], 'multi':[2,6,2*l,6*l]}
    ep_range[3] = {'good':[2,6,8], 'multi':[2,6,2*l,6*l]}
    ep_range[l] = {'good':[l-1,l*(l-1),l*l-1], 'multi':[l-1,3*(l-1),l*(l-1),3*l*(l-1)]}
    return ep_range

def compute_log_volume(get_ramification_dataset, e0, l):
    if l < 5:
        raise ValueError("The prime number l should be >= 5")
    ep_range = get_ramification_dataset(l)
    my_primes = primes(e0 * l + 1)
    vol = log(pi)
    for p in my_primes:
        vols_at_p = []
        if p in ep_range: # special case
            p_key = p
        else:
            p_key = 'general' # general case
        vols_at_p = []
        for ep in ep_range[p_key]['good']:
            vols_at_p.append(log_volume_at_p(l, p, ep, e0, 'good'))
        for ep in ep_range[p_key]['multi']:
            vols_at_p.append(log_volume_at_p(l, p, ep, e0, 'multi'))
        vol += max(vols_at_p) * log(p)
    vol *= Fraction(l*l+5*l, l*l+l-12)
    return max(vol,0)

# fast version
def log_volume_at_p_fast(l, p, ep, e0, reduction_type):
    # compute the upper bound of "certain contribution of p in Vol(l)" for given ep
    l1 = l // 2
    upper_bound = 0
    if reduction_type == 'good':
        is_multi = 0
    elif reduction_type == 'multi':
        is_multi = 1
    else:
        raise ValueError('reduction type invalid')
    # p-adic valuation of ep
    vp_ep = compute_valuation(ep, p)
    # dp
    if vp_ep == 0:
        dp = 1 - Fraction(1, ep)
    else:
        dp = vp_ep + 1
    if ep==2 and p==2:
        dp = 1
    # ap = (ep // (p-1) + 1) / ep, bp = max{n - p^n / ep}
    ap = Fraction(ep//(p-1) + 1, ep)
    if ep <= p-1:
        bp = Fraction(-1,ep)
    elif ep <= p*(p-1):
        bp = 1-Fraction(p,ep)
    else:
        n0 = ceil(log(Fraction(ep,p-1))/log(p))
        bp = n0 - Fraction(pow(p,n0),ep)

    # the contribution at vp in \log\rad(N)
    if ep % l != 0:
        cp = is_multi * (1 - Fraction(1, e0))         
    else: # if ep % l == 0:
        cp = is_multi * (1 - Fraction(1, e0*l))
        

    # the upper bound of B_1(p)
    for j in range(1,l1+1):
        upper_bound += max((j*dp+(j+1)*ap + 1 - Fraction(1,ep)), (j+1)*(1+(p==2)))+ (j+1)*bp
    upper_bound = Fraction(upper_bound, l1) / Fraction(l+5, 4) - cp
    return upper_bound

def compute_log_volume_fast(get_ramification_dataset, e0, l):
    if l < 5:
        raise ValueError("The prime number l should be >= 5")
    ep_range = get_ramification_dataset(l)
    my_primes = primes(e0 * l + 1)
    vol = log(pi)
    for p in my_primes:
        vols_at_p = []
        if p in ep_range: # special case
            p_key = p
        else:
            p_key = 'general' # general case
        vols_at_p = []
        for ep in ep_range[p_key]['good']:
            vols_at_p.append(log_volume_at_p_fast(l, p, ep, e0, 'good'))
        for ep in ep_range[p_key]['multi']:
            vols_at_p.append(log_volume_at_p_fast(l, p, ep, e0, 'multi'))
        vol += max(vols_at_p) * log(p)
    vol *= Fraction(l*l+5*l, l*l+l-12)
    return max(vol,0)
