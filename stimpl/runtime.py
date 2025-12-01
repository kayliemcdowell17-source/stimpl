from typing import Any, Tuple, Optional

from stimpl.expression import (
    Program, Sequence, Assign, Variable, Ren, Print,
    IntLiteral, FloatingPointLiteral, StringLiteral, BooleanLiteral,
    Add, Subtract, Multiply, Divide,
    And, Or, Not,
    Lt, Lte, Gt, Gte, Eq, Ne,
    If, While
)
from stimpl.types import Integer, FloatingPoint, String, Boolean, Unit
from stimpl.errors import InterpTypeError, InterpSyntaxError, InterpMathError

"""
Interpreter State
"""


class State(object):
    def __init__(self, variable_name: str = None, variable_value: Any = None, variable_type: Any = None, next_state: 'State' = None) -> None:
        self.variable_name = variable_name
        self.value = (variable_value, variable_type)
        self.next_state = next_state

    def copy(self) -> 'State':
        variable_value, variable_type = self.value
        return State(self.variable_name, variable_value, variable_type, self.next_state)

    def set_value(self, variable_name: str, variable_value: Any, variable_type: Any):
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
        super().__init__(None, None, None, None)

    def copy(self) -> 'EmptyState':
        return EmptyState()

    def get_value(self, variable_name: str) -> None:
        return None

    def __repr__(self) -> str:
        return ""


"""
Main evaluation logic!
"""


def evaluate(expression: Any, state: State) -> Tuple[Optional[Any], Any, State]:
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
            value, typ, new_state = evaluate(to_print, state)
            print(value if not isinstance(typ, Unit) else "Unit")
            return (value, typ, new_state)

        case Sequence(exprs=exprs) | Program(exprs=exprs):
            last_value, last_type, current_state = None, Unit(), state
            for expr in exprs:
                last_value, last_type, current_state = evaluate(expr, current_state)
            return (last_value, last_type, current_state)

        case Variable(variable_name=variable_name):
            value = state.get_value(variable_name)
            if value is None:
                raise InterpSyntaxError(
                    f"Cannot read from {variable_name} before assignment.")
            variable_value, variable_type = value
            return (variable_value, variable_type, state)

        case Assign(variable=variable, value=value):
            value_result, value_type, new_state = evaluate(value, state)
            var_val = new_state.get_value(variable.variable_name)
            _, var_type = var_val if var_val else (None, None)

            if var_type is not None and var_type != value_type:
                raise InterpTypeError(
                    f"Mismatched types for Assignment: Cannot assign {value_type} to {var_type}")

            updated_state = new_state.set_value(
                variable.variable_name, value_result, value_type)
            return (value_result, value_type, updated_state)

        case Add(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)

            if left_type != right_type:
                raise InterpTypeError(f"Cannot add {left_type} to {right_type}")

            match left_type:
                case Integer() | FloatingPoint() | String():
                    return (left_val + right_val, left_type, new_state)
                case _:
                    raise InterpTypeError(f"Cannot add {left_type}")

        case Subtract(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot subtract {left_type} and {right_type}")
            match left_type:
                case Integer() | FloatingPoint():
                    return (left_val - right_val, left_type, new_state)
                case _:
                    raise InterpTypeError(f"Cannot subtract {left_type}")

        case Multiply(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot multiply {left_type} and {right_type}")
            match left_type:
                case Integer() | FloatingPoint():
                    return (left_val * right_val, left_type, new_state)
                case _:
                    raise InterpTypeError(f"Cannot multiply {left_type}")

        case Divide(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot divide {left_type} and {right_type}")
            if right_val == 0 or right_val == 0.0:
                raise InterpMathError("Division by zero")
            match left_type:
                case Integer():
                    return (left_val // right_val, Integer(), new_state)
                case FloatingPoint():
                    return (left_val / right_val, FloatingPoint(), new_state)
                case _:
                    raise InterpTypeError(f"Cannot divide {left_type}")

        case And(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, Boolean):
                raise InterpTypeError(f"Cannot perform And on non-boolean")
            return (left_val and right_val, Boolean(), new_state)

        case Or(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, Boolean):
                raise InterpTypeError(f"Cannot perform Or on non-boolean")
            return (left_val or right_val, Boolean(), new_state)

        case Not(expr=expr):
            val, typ, new_state = evaluate(expr, state)
            if not isinstance(typ, Boolean):
                raise InterpTypeError("Cannot perform Not on non-boolean")
            return (not val, Boolean(), new_state)

        case If(condition=condition, true=true, false=false):
            cond_val, cond_type, new_state = evaluate(condition, state)
            if not isinstance(cond_type, Boolean):
                raise InterpTypeError("If condition must be Boolean")
            if cond_val:
                return evaluate(true, new_state)
            else:
                return evaluate(false, new_state)

        case Lt(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot compare {left_type} and {right_type}")
            if isinstance(left_type, (Integer, FloatingPoint, String, Boolean)):
                return (left_val < right_val, Boolean(), new_state)
            if isinstance(left_type, Unit):
                return (False, Boolean(), new_state)
            raise InterpTypeError(f"Cannot perform < on {left_type}")

        case Lte(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot compare {left_type} and {right_type}")
            if isinstance(left_type, (Integer, FloatingPoint, String, Boolean)):
                return (left_val <= right_val, Boolean(), new_state)
            if isinstance(left_type, Unit):
                return (True, Boolean(), new_state)
            raise InterpTypeError(f"Cannot perform <= on {left_type}")

        case Gt(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot compare {left_type} and {right_type}")
            if isinstance(left_type, (Integer, FloatingPoint, String, Boolean)):
                return (left_val > right_val, Boolean(), new_state)
            if isinstance(left_type, Unit):
                return (False, Boolean(), new_state)
            raise InterpTypeError(f"Cannot perform > on {left_type}")

        case Gte(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot compare {left_type} and {right_type}")
            if isinstance(left_type, (Integer, FloatingPoint, String, Boolean)):
                return (left_val >= right_val, Boolean(), new_state)
            if isinstance(left_type, Unit):
                return (True, Boolean(), new_state)
            raise InterpTypeError(f"Cannot perform >= on {left_type}")

        case Eq(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot compare {left_type} and {right_type}")
            return (left_val == right_val, Boolean(), new_state)

        case Ne(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Cannot compare {left_type} and {right_type}")
            return (left_val != right_val, Boolean(), new_state)

        case While(condition=condition, body=body):
            cond_val, cond_type, new_state = evaluate(condition, state)
            if not isinstance(cond_type, Boolean):
                raise InterpTypeError("While condition must be Boolean")
            current_state = new_state
            while cond_val:
                _, _, current_state = evaluate(body, current_state)
                cond_val, cond_type, current_state = evaluate(condition, current_state)
            return (False, Boolean(), current_state)

        case _:
            raise InterpSyntaxError("Unhandled expression")


def run_stimpl(program, debug=False):
    state = EmptyState()
    program_value, program_type, program_state = evaluate(program, state)

    if debug:
        print(f"program: {program}")
        print(f"final_value: ({program_value}, {program_type})")
        print(f"final_state: {program_state}")

    return program_value, program_type, program_state
