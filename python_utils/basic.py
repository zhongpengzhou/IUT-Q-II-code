from math import sqrt,log,floor,ceil,exp,gcd,pi,prod
from tqdm import trange,tqdm
from fractions import Fraction


def is_prime(n):
    if n < 2:
        return False
    for i in range(2,n):
        if n%i == 0:
            return False
        if i * i > n:
            break
    return True

def primes(n):
    if n < 2:
        return []
    res = list()
    sieve = [True] * (n+1)
    for p in range(2, n+1):
        if (sieve[p]):
            res.append(p)
            for i in range(2*p, n+1, p):
                sieve[i] = False
    return res
            
def prime_counting(x):
    return len(primes(int(x)))

def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][-1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][-1] = max(merged[-1][-1], interval[-1])
    return merged

def divisors(S):
    # find the list of divisors of an positive integer or a list/tuple of positive integers S
    divisors = set() 
    if type(S) != int:
        for k in S:
           divisors.update(divisors(k))
        return sorted(divisors)
    for i in range(1, int(sqrt(S)) + 1):
        if S % i == 0:
            divisors.add(i)
            divisors.add(S // i)
    return sorted(divisors)

def compute_valuation(n,p):
    if n == 0:
        raise ValueError("cannot compute the valuation of zero")
    res = 0
    while(n % p == 0):
        res += 1
        n = n // p
    return res