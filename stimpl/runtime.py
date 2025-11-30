from typing import Any, Tuple, Optional

from stimpl.expression import *
from stimpl.types import *
from stimpl.errors import *

"""
Interpreter State
"""


class State(object):
    def __init__(self, variable_name: str, variable_value: Expr, variable_type: Type, next_state: 'State') -> None:
        self.variable_name = variable_name
        self.value = (variable_value, variable_type)
        self.next_state = next_state

    def copy(self) -> 'State':
        variable_value, variable_type = self.value
        return State(self.variable_name, variable_value, variable_type, self.next_state)

    def set_value(self, variable_name: str, variable_value: Expr, variable_type: Type):
        return State(variable_name, variable_value, variable_type, self)

    def get_value(self, variable_name: str) -> Any:
        """Walk linked-list environment and return (value, type) or None."""
        cur = self
        while isinstance(cur, State):
            if cur.variable_name == variable_name:
                return cur.value
            cur = cur.next_state
        return None

    def __repr__(self) -> str:
        return f"{self.variable_name}: {self.value}, " + repr(self.next_state)


class EmptyState(State):
    def __init__(self):
        pass

    def copy(self) -> 'EmptyState':
        return EmptyState()

    def get_value(self, variable_name: str) -> None:
        return None

    def __repr__(self) -> str:
        return ""


"""
Main evaluation logic!
"""


def evaluate(expression: Expr, state: State) -> Tuple[Optional[Any], Type, State]:
    match expression:

        case Ren():
            return ((), Unit(), state)

        case IntLiteral(literal=l):
            return (l, Integer(), state)

        case FloatingPointLiteral(literal=l):
            return (l, FloatingPoint(), state)

        case StringLiteral(literal=l):
            return (l, String(), state)

        case BooleanLiteral(literal=l):
            return (l, Boolean(), state)

        case Print(to_print=to_print):
            printable_value, printable_type, new_state = evaluate(to_print, state)

            match printable_type:
                case Unit():
                    print("Unit")
                case _:
                    print(f"{printable_value}")

            return (printable_value, printable_type, new_state)

        # -------------------------
        # Sequence / Program
        # -------------------------
        case Sequence(exprs=exprs) | Program(exprs=exprs):
            new_state = state
            last_val, last_type = None, Unit()
            if len(exprs) == 0:
                return (None, Unit(), state)

            for e in exprs:
                last_val, last_type, new_state = evaluate(e, new_state)

            return (last_val, last_type, new_state)

        # -------------------------
        # Variable Read
        # -------------------------
        case Variable(variable_name=variable_name):
            value = state.get_value(variable_name)
            if value is None:
                raise InterpSyntaxError(
                    f"Cannot read from {variable_name} before assignment."
                )
            variable_value, variable_type = value
            return (variable_value, variable_type, state)

        # -------------------------
        # Assignment
        # -------------------------
        case Assign(variable=variable, value=value):

            value_result, value_type, new_state = evaluate(value, state)

            variable_from_state = new_state.get_value(variable.variable_name)
            _, variable_type = variable_from_state if variable_from_state else (None, None)

            if value_type != variable_type and variable_type != None:
                raise InterpTypeError(f"""Mismatched types for Assignment:
            Cannot assign {value_type} to {variable_type}""")

            new_state = new_state.set_value(
                variable.variable_name, value_result, value_type
            )
            return (value_result, value_type, new_state)

        # -------------------------
        # Add (already implemented)
        # -------------------------
        case Add(left=left, right=right):
            result = 0
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)

            if left_type != right_type:
                raise InterpTypeError(f"""Mismatched types for Add:
            Cannot add {left_type} to {right_type}""")

            match left_type:
                case Integer() | String() | FloatingPoint():
                    result = left_result + right_result
                case _:
                    raise InterpTypeError(f"""Cannot add {left_type}s""")

            return (result, left_type, new_state)

        # -------------------------
        # SUBTRACT
        # -------------------------
        case Subtract(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError(f"Mismatched types for Subtract: {lt} vs {rt}")

            match lt:
                case Integer() | FloatingPoint():
                    return (l - r, lt, s2)
                case _:
                    raise InterpTypeError("Cannot subtract non-numeric types.")

        # -------------------------
        # MULTIPLY
        # -------------------------
        case Multiply(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError(f"Mismatched types for Multiply: {lt} vs {rt}")

            match lt:
                case Integer() | FloatingPoint():
                    return (l * r, lt, s2)
                case _:
                    raise InterpTypeError("Cannot multiply non-numeric types.")

        # -------------------------
        # DIVIDE
        # -------------------------
        case Divide(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError(f"Mismatched types for Divide: {lt} vs {rt}")

            match lt:
                case Integer():
                    if r == 0:
                        raise InterpMathError("Division by zero.")
                    return (l // r, lt, s2)
                case FloatingPoint():
                    if r == 0.0:
                        raise InterpMathError("Division by zero.")
                    return (l / r, lt, s2)
                case _:
                    raise InterpTypeError("Cannot divide non-numeric types.")

        # -------------------------
        # AND (already implemented)
        # -------------------------

        # -------------------------
        # OR
        # -------------------------
        case Or(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt or not isinstance(lt, Boolean):
                raise InterpTypeError("Cannot perform OR on non-boolean types.")

            return (l or r, lt, s2)

        # -------------------------
        # NOT
        # -------------------------
        case Not(expr=expr):
            v, t, s1 = evaluate(expr, state)

            if not isinstance(t, Boolean):
                raise InterpTypeError("Cannot perform NOT on non-boolean type.")

            return (not v, t, s1)

        # -------------------------
        # IF
        # -------------------------
        case If(condition=condition, true=true, false=false):
            cval, ctype, s1 = evaluate(condition, state)

            if not isinstance(ctype, Boolean):
                raise InterpTypeError("Condition of IF must be boolean.")

            if cval:
                return evaluate(true, s1)
            else:
                return evaluate(false, s1)

        # -------------------------
        # COMPARISONS (<=, >, >=)
        # -------------------------
        case Lte(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError("Mismatched types in <=")

            match lt:
                case Integer() | FloatingPoint() | Boolean() | String():
                    return (l <= r, Boolean(), s2)
                case Unit():
                    return (True, Boolean(), s2)
                case _:
                    raise InterpTypeError("Invalid type for <=")

        case Gt(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError("Mismatched types in >")

            match lt:
                case Integer() | FloatingPoint() | Boolean() | String():
                    return (l > r, Boolean(), s2)
                case Unit():
                    return (False, Boolean(), s2)
                case _:
                    raise InterpTypeError("Invalid type for >")

        case Gte(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError("Mismatched types in >=")

            match lt:
                case Integer() | FloatingPoint() | Boolean() | String():
                    return (l >= r, Boolean(), s2)
                case Unit():
                    return (True, Boolean(), s2)
                case _:
                    raise InterpTypeError("Invalid type for >=")

        # -------------------------
        # EQ / NE
        # -------------------------
        case Eq(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError("Mismatched types in ==")

            return (l == r, Boolean(), s2)

        case Ne(left=left, right=right):
            l, lt, s1 = evaluate(left, state)
            r, rt, s2 = evaluate(right, s1)

            if lt != rt:
                raise InterpTypeError("Mismatched types in !=")

            return (l != r, Boolean(), s2)

        # -------------------------
        # WHILE
        # -------------------------
        case While(condition=condition, body=body):
            new_state = state

            while True:
                cond_val, cond_type, new_state = evaluate(condition, new_state)

                if not isinstance(cond_type, Boolean):
                    raise InterpTypeError("Condition of WHILE must be boolean.")

                if not cond_val:
                    return (False, Boolean(), new_state)

                _, _, new_state = evaluate(body, new_state)

        # -------------------------
        # FALLTHROUGH
        # -------------------------
        case _:
            raise InterpSyntaxError("Unhandled expression.")

    pass


def run_stimpl(program, debug=False):
    state = EmptyState()
    program_value, program_type, program_state = evaluate(program, state)

    if debug:
        print(f"program: {program}")
        print(f"final_value: ({program_value}, {program_type})")
        print(f"final_state: {program_state}")

    return program_value, program_type, program_state
