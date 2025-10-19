from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Union

TOK_DIGITS = list("0123456789")
TOK_VARS = ["x","y","z"]
OP_ALL = ["+","-","·","*","÷","="]

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
    exp: "Expr"   # single symbol (digit or var)

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
    if depth <= 0:
        return base
    if random.random() < 0.25:
        exp: Expr = random_var() if random.random() < 0.5 else random_digit()
        return Pow(base=base, exp=exp)
    return base

def frac_or_term(depth: int) -> Expr:
    if depth > 0 and random.random() < 0.18:
        return Frac(num=arith_expr(depth-1, allowed_ops=["+","-","·","*","÷"]), den=arith_expr(depth-1, allowed_ops=["+","-","·","*","÷"]))
    base = atom(depth)
    return maybe_pow(base, depth)

def chain_ops(left: Expr, depth: int, max_additional: int, allowed_ops: List[str]) -> Expr:
    node = left
    t = random.randint(0, max_additional)
    for _ in range(t):
        op = random.choice(allowed_ops)
        right = frac_or_term(depth)
        node = BinOp(op=op, left=node, right=right)
    return node

def arith_expr(depth: int, allowed_ops: List[str]) -> Expr:
    return chain_ops(frac_or_term(depth), depth, max_additional=2, allowed_ops=allowed_ops)

def equality_expr(depth: int, allowed_ops: List[str]) -> Expr:
    left = arith_expr(depth, allowed_ops)
    right = arith_expr(depth, allowed_ops)
    return BinOp(op="=", left=left, right=right)

def gen_expression(max_depth: int = 2, eq_prob: float = 0.25, allowed_ops: List[str] = None) -> Expr:
    if allowed_ops is None:
        allowed_ops = ["+","-","·","*","÷"]
    if random.random() < eq_prob:
        return equality_expr(max_depth, allowed_ops)
    return arith_expr(max_depth, allowed_ops)

def serialize(expr: Expr) -> List[str]:
    if isinstance(expr, Num):
        return list(expr.s)
    if isinstance(expr, Var):
        return [expr.s]
    if isinstance(expr, BinOp):
        return serialize(expr.left) + [expr.op] + serialize(expr.right)
    if isinstance(expr, Pow):
        return serialize(expr.base) + ["^","("] + serialize(expr.exp) + [")"]
    if isinstance(expr, Frac):
        return ["f","r","a","c","(","{"] + serialize(expr.num) + ["}",",","{"] + serialize(expr.den) + ["}",")"]
    raise TypeError("unknown expr type")
