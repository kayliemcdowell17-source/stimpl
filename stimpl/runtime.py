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
            printable_value, printable_type, new_state = evaluate(to_print, state)
            if isinstance(printable_type, Unit):
                print("Unit")
            else:
                print(printable_value)
            return (printable_value, printable_type, new_state)

        case Sequence(exprs=exprs) | Program(exprs=exprs):
            if len(exprs) == 0:
                return (None, Unit(), state)
            current_state = state
            result_value = None
            result_type = Unit()
            for expr in exprs:
                result_value, result_type, current_state = evaluate(expr, current_state)
            return (result_value, result_type, current_state)

        case Variable(variable_name=variable_name):
            value = state.get_value(variable_name)
            if value is None:
                raise InterpSyntaxError(f"Cannot read from {variable_name} before assignment.")
            variable_value, variable_type = value
            return (variable_value, variable_type, state)

        case Assign(variable=variable, value=value):
            value_result, value_type, new_state = evaluate(value, state)
            variable_from_state = new_state.get_value(variable.variable_name)
            _, variable_type = variable_from_state if variable_from_state else (None, None)
            if variable_type is not None and value_type != variable_type:
                raise InterpTypeError(f"Mismatched types for Assignment: Cannot assign {value_type} to {variable_type}")
            new_state = new_state.set_value(variable.variable_name, value_result, value_type)
            return (value_result, value_type, new_state)

        case Add(left=left, right=right):
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Mismatched types for Add: Cannot add {left_type} to {right_type}")
            if isinstance(left_type, (Integer, FloatingPoint, String)):
                result = left_result + right_result
            else:
                raise InterpTypeError(f"Cannot add {left_type}s")
            return (result, left_type, new_state)

        case Subtract(left=left, right=right):
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, (Integer, FloatingPoint)):
                raise InterpTypeError(f"Cannot subtract {right_type} from {left_type}")
            return (left_result - right_result, left_type, new_state)

        case Multiply(left=left, right=right):
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, (Integer, FloatingPoint)):
                raise InterpTypeError(f"Cannot multiply {left_type} with {right_type}")
            return (left_result * right_result, left_type, new_state)

        case Divide(left=left, right=right):
            left_result, left_type, new_state = evaluate(left, state)
            right_result, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, (Integer, FloatingPoint)):
                raise InterpTypeError(f"Cannot divide {left_type} by {right_type}")
            if right_result == 0:
                raise InterpMathError("Division by zero")
            if isinstance(left_type, Integer):
                return (left_result // right_result, left_type, new_state)
            else:
                return (left_result / right_result, left_type, new_state)

        case And(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, Boolean):
                raise InterpTypeError("And operator requires boolean operands")
            return (left_val and right_val, Boolean(), new_state)

        case Or(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type or not isinstance(left_type, Boolean):
                raise InterpTypeError("Or operator requires boolean operands")
            return (left_val or right_val, Boolean(), new_state)

        case Not(expr=expr):
            val, val_type, new_state = evaluate(expr, state)
            if not isinstance(val_type, Boolean):
                raise InterpTypeError("Not operator requires a boolean operand")
            return (not val, Boolean(), new_state)

        case If(condition=condition, true=true, false=false):
            cond_val, cond_type, new_state = evaluate(condition, state)
            if not isinstance(cond_type, Boolean):
                raise InterpTypeError("If condition must be boolean")
            if cond_val:
                return evaluate(true, new_state)
            else:
                return evaluate(false, new_state)

        case Lt(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError(f"Mismatched types for Lt: {left_type} vs {right_type}")
            if isinstance(left_type, (Integer, FloatingPoint, String, Boolean)):
                return (left_val < right_val, Boolean(), new_state)
            if isinstance(left_type, Unit):
                return (False, Boolean(), new_state)
            raise InterpTypeError(f"Cannot perform Lt on {left_type}")

        case Lte(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError("Mismatched types for Lte")
            return (left_val <= right_val, Boolean(), new_state)

        case Gt(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError("Mismatched types for Gt")
            return (left_val > right_val, Boolean(), new_state)

        case Gte(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError("Mismatched types for Gte")
            return (left_val >= right_val, Boolean(), new_state)

        case Eq(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError("Mismatched types for Eq")
            return (left_val == right_val, Boolean(), new_state)

        case Ne(left=left, right=right):
            left_val, left_type, new_state = evaluate(left, state)
            right_val, right_type, new_state = evaluate(right, new_state)
            if left_type != right_type:
                raise InterpTypeError("Mismatched types for Ne")
            return (left_val != right_val, Boolean(), new_state)

        case While(condition=condition, body=body):
            current_state = state
            cond_val, cond_type, current_state = evaluate(condition, current_state)
            if not isinstance(cond_type, Boolean):
                raise InterpTypeError("While condition must be boolean")
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
