from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Union

TOK_DIGITS = list("0123456789")
TOK_VARS = ["x","y","z"]
OP_ARITH = ["+","-","*","÷"]
OP_EQ = "="

@dataclass
class Num:
    s: str

@dataclass
class Var:
    s: str

@dataclass
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"

@dataclass
class Pow:
    base: "Expr"
    exp: "Expr"   # now ALWAYS a single symbol (digit or var)

@dataclass
class Frac:
    num: "Expr"
    den: "Expr"

Expr = Union[Num, Var, BinOp, Pow, Frac]

def random_number(max_len: int = 3) -> Num:
    L = random.randint(1, max_len)
    s = "".join(random.choice(TOK_DIGITS) for _ in range(L))
    if len(s) > 1 and s[0] == "0":
        s = str(random.randint(1,9)) + s[1:]
    return Num(s)

def random_digit() -> Num:
    return Num(random.choice(TOK_DIGITS))

def random_var() -> Var:
    return Var(random.choice(TOK_VARS))

def atom(depth: int) -> Expr:
    return random_number() if random.random() < 0.6 else random_var()

def maybe_pow(base: Expr, depth: int) -> Expr:
    # New rule: exponent is strictly ONE symbol (digit or variable)
    if depth <= 0:
        return base
    if random.random() < 0.25:
        exp: Expr = random_var() if random.random() < 0.5 else random_digit()
        return Pow(base=base, exp=exp)
    return base

def frac_or_term(depth: int) -> Expr:
    if depth > 0 and random.random() < 0.18:
        return Frac(num=arith_expr(depth-1), den=arith_expr(depth-1))
    base = atom(depth)
    return maybe_pow(base, depth)

def chain_ops(left: Expr, depth: int, max_additional: int) -> Expr:
    node = left
    t = random.randint(0, max_additional)
    for _ in range(t):
        op = random.choice(OP_ARITH)
        right = frac_or_term(depth)
        node = BinOp(op=op, left=node, right=right)
    return node

def arith_expr(depth: int) -> Expr:
    return chain_ops(frac_or_term(depth), depth, max_additional=2)

def equality_expr(depth: int) -> Expr:
    left = arith_expr(depth)
    right = arith_expr(depth)
    return BinOp(op=OP_EQ, left=left, right=right)

def gen_expression(max_depth: int = 2, eq_prob: float = 0.25) -> Expr:
    return equality_expr(max_depth) if random.random() < eq_prob else arith_expr(max_depth)

def serialize(expr: Expr) -> List[str]:
    if isinstance(expr, Num):
        return list(expr.s)
    if isinstance(expr, Var):
        return [expr.s]
    if isinstance(expr, BinOp):
        return serialize(expr.left) + [expr.op] + serialize(expr.right)
    if isinstance(expr, Pow):
        # Keep canonical '^ ( token )' even though token is single symbol
        return serialize(expr.base) + ["^","("] + serialize(expr.exp) + [")"]
    if isinstance(expr, Frac):
        return ["f","r","a","c","(","{"] + serialize(expr.num) + ["}",",","{"] + serialize(expr.den) + ["}",")"]
    raise TypeError("unknown expr type")
