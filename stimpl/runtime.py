from typing import Any, Tuple, Optional

from stimpl.expression import (
    Expr, Program, Sequence, Assign, Variable, Ren, Print,
    IntLiteral, FloatingPointLiteral, StringLiteral, BooleanLiteral,
    Add, Subtract, Multiply, Divide,
    And, Or, Not,
    Lt, Lte, Gt, Gte, Eq, Ne,
    If, While
)
from stimpl.types import Integer, FloatingPoint, String, Boolean, Unit, Type
from stimpl.errors import InterpTypeError, InterpSyntaxError, InterpMathError


class State(object):
    def __init__(self, variable_name=None, variable_value=None, variable_type=None, next_state=None):
        self.variable_name = variable_name
        self.value = (variable_value, variable_type)
        self.next_state = next_state

    def copy(self) -> 'State':
        variable_value, variable_type = self.value
        return State(self.variable_name, variable_value, variable_type, self.next_state)

    def set_value(self, variable_name: str, variable_value: Any, variable_type: Type):
        return State(variable_name, variable_value, variable_type, self)

    def get_value(self, variable_name: str) -> Any:
        state = self
        while state is not None:
            if state.variable_name == variable_name:
                return state.value
            state = state.next_state
        return None

    def __repr__(self) -> str:
        return f"{self.variable_name}: {self.value}, " + repr(self.next_state)


class EmptyState(State):
    def __init__(self):
        super().__init__()

    def copy(self) -> 'EmptyState':
        return EmptyState()

    def get_value(self, variable_name: str) -> None:
        return None

    def __repr__(self) -> str:
        return ""


def evaluate(expression: Expr, state: State) -> Tuple[Optional[Any], Type, State]:
    match expression:
        case Ren():
            return (None, Unit(), state)

        case IntLiteral(literal=l):
            return (l, Integer(), state)

        case FloatingPointLiteral(literal=l):
            return (l, FloatingPoint(), state)

        case StringLiteral(literal=l):
            return (l, String(), state)

        case BooleanLiteral(literal=l):
            return (l, Boolean(), state)

        case Print(to_print=to_print):
            value, typ, new_state = evaluate(to_print, state)
            if isinstance(value, Unit):
                print("Unit")
            else:
                print(value)
            return (value, typ, new_state)

        case Sequence(exprs=exprs) | Program(exprs=exprs):
            current_state = state
            result = None
            result_type = Unit()
            for expr in exprs:
                result, result_type, current_state = evaluate(expr, current_state)
            return (result, result_type, current_state)

        case Variable(variable_name=variable_name):
            value = state.get_value(variable_name)
            if value is None:
                raise InterpSyntaxError(f"Cannot read from {variable_name} before assignment.")
            variable_value, variable_type = value
            return (variable_value, variable_type, state)

        case Assign(variable=variable, value=value):
            value_result, value_type, new_state = evaluate(value, state)
            current_value = new_state.get_value(variable.variable_name)
            _, variable_type = current_value if current_value else (None, None)
            if variable_type is not None and value_type != variable_type:
                raise InterpTypeError(f"Mismatched types for Assignment: Cannot assign {value_type} to {variable_type}")
            updated_state = new_state.set_value(variable.variable_name, value_result, value_type)
            return (value_result, value_type, updated_state)

        case Add(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Cannot add {l_type} to {r_type}")
            if isinstance(l_type, (Integer, FloatingPoint, String)):
                return (l_val + r_val, l_type, s2)
            raise InterpTypeError(f"Cannot add {l_type} types")

        case Subtract(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type or not isinstance(l_type, (Integer, FloatingPoint)):
                raise InterpTypeError(f"Cannot subtract {r_type} from {l_type}")
            return (l_val - r_val, l_type, s2)

        case Multiply(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type or not isinstance(l_type, (Integer, FloatingPoint)):
                raise InterpTypeError(f"Cannot multiply {l_type} and {r_type}")
            return (l_val * r_val, l_type, s2)

        case Divide(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type or not isinstance(l_type, (Integer, FloatingPoint)):
                raise InterpTypeError(f"Cannot divide {l_type} by {r_type}")
            if r_val == 0:
                raise InterpMathError("Division by zero")
            if isinstance(l_type, Integer):
                return (l_val // r_val, l_type, s2)
            return (l_val / r_val, l_type, s2)

        case And(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type or not isinstance(l_type, Boolean):
                raise InterpTypeError(f"Cannot perform logical AND on {l_type} and {r_type}")
            return (l_val and r_val, l_type, s2)

        case Or(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type or not isinstance(l_type, Boolean):
                raise InterpTypeError(f"Cannot perform logical OR on {l_type} and {r_type}")
            return (l_val or r_val, l_type, s2)

        case Not(expr=expr):
            val, typ, new_state = evaluate(expr, state)
            if not isinstance(typ, Boolean):
                raise InterpTypeError(f"Cannot perform NOT on {typ}")
            return (not val, typ, new_state)

        case If(condition=condition, true=true, false=false):
            cond_val, cond_type, new_state = evaluate(condition, state)
            if not isinstance(cond_type, Boolean):
                raise InterpTypeError(f"Condition must be boolean, got {cond_type}")
            branch = true if cond_val else false
            return evaluate(branch, new_state)

        case Lt(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Mismatched types for Lt: {l_type}, {r_type}")
            if isinstance(l_type, (Integer, FloatingPoint, String, Boolean)):
                return (l_val < r_val, Boolean(), s2)
            if isinstance(l_type, Unit):
                return (False, Boolean(), s2)
            raise InterpTypeError(f"Cannot perform < on {l_type}")

        case Lte(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Mismatched types for Lte: {l_type}, {r_type}")
            if isinstance(l_type, (Integer, FloatingPoint, String, Boolean)):
                return (l_val <= r_val, Boolean(), s2)
            if isinstance(l_type, Unit):
                return (True, Boolean(), s2)
            raise InterpTypeError(f"Cannot perform <= on {l_type}")

        case Gt(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Mismatched types for Gt: {l_type}, {r_type}")
            if isinstance(l_type, (Integer, FloatingPoint, String, Boolean)):
                return (l_val > r_val, Boolean(), s2)
            if isinstance(l_type, Unit):
                return (False, Boolean(), s2)
            raise InterpTypeError(f"Cannot perform > on {l_type}")

        case Gte(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Mismatched types for Gte: {l_type}, {r_type}")
            if isinstance(l_type, (Integer, FloatingPoint, String, Boolean)):
                return (l_val >= r_val, Boolean(), s2)
            if isinstance(l_type, Unit):
                return (True, Boolean(), s2)
            raise InterpTypeError(f"Cannot perform >= on {l_type}")

        case Eq(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Mismatched types for Eq: {l_type}, {r_type}")
            return (l_val == r_val, Boolean(), s2)

        case Ne(left=left, right=right):
            l_val, l_type, s1 = evaluate(left, state)
            r_val, r_type, s2 = evaluate(right, s1)
            if l_type != r_type:
                raise InterpTypeError(f"Mismatched types for Ne: {l_type}, {r_type}")
            return (l_val != r_val, Boolean(), s2)

        case While(condition=condition, body=body):
            current_state = state
            cond_val, cond_type, current_state = evaluate(condition, current_state)
            if not isinstance(cond_type, Boolean):
                raise InterpTypeError("While loop condition must be boolean")
            while cond_val:
                _, _, current_state = evaluate(body, current_state)
                cond_val, cond_type, current_state = evaluate(condition, current_state)
            return (False, Boolean(), current_state)

        case _:
            raise InterpSyntaxError("Unhandled expression type!")


def run_stimpl(program: Expr, debug=False):
    state = EmptyState()
    program_value, program_type, program_state = evaluate(program, state)

    if debug:
        print(f"program: {program}")
        print(f"final_value: ({program_value}, {program_type})")
        print(f"final_state: {program_state}")

    return program_value, program_type, program_state
