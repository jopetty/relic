import fire
import numpy as np
import os

import random
from nltk import PCFG

import glob
import ast
import operator
from tqdm import tqdm, trange


CRASP_MIN_GRAMMAR = """
S -> FUNC [1.0]

FUNC -> 'lambda' 'x, y, z, arr' ':' EXPR [1.0]

EXPR -> EXPR '+' TERM [0.4]  | TERM [0.3] |  MIN_MAX [0.1] | COUNT [0.1] | IF_ELSE [0.1]
                        
EXPR_NO_IE -> EXPR_NO_IE '+' TERM [0.4]  | TERM [0.3] |  MIN_MAX [0.2] | COUNT [0.1] 
                        
TERM -> VAR [0.4] | INTEGER [0.2] | '(' EXPR ')' [0.4]

MIN_MAX -> 'min(' EXPR ',' EXPR ')' [1.0] 

COUNT -> 'arr.count(' INTEGER ')' [1.0]

BOOLEAN -> 'not' BOOLEAN [0.10] | BOOLEAN 'and' BOOLEAN [0.10] | BOOLEAN 'or' BOOLEAN [0.10] | EXPR_NO_IE '==' EXPR_NO_IE [0.10] | EXPR_NO_IE '!=' EXPR_NO_IE [0.10] | EXPR_NO_IE '<' EXPR_NO_IE [0.10] | EXPR_NO_IE '<=' EXPR_NO_IE [0.10] | EXPR_NO_IE '>' EXPR_NO_IE [0.10] | EXPR_NO_IE '>=' EXPR_NO_IE [0.10] | 'True' [0.05] | 'False' [0.05]

IF_ELSE -> EXPR_NO_IE 'if' BOOLEAN 'else' EXPR [1.0]

INTEGER -> '0' [0.1] | '1' [0.1] | '2' [0.1] | '3' [0.1] | '4' [0.1] | '5' [0.1] | '6' [0.1] | '7' [0.1] | '8' [0.1] | '9' [0.1]

VAR -> 'x' [0.4] | 'y' [0.3] | 'z' [0.3]"""

CRASP_GRAMMAR = """
S -> FUNC [1.0]

FUNC -> 'lambda' 'x, y, z, arr' ':' EXPR [1.0]

EXPR -> EXPR '+' TERM [0.25] | EXPR '-' TERM [0.25] | TERM [0.2] |  MIN_MAX [0.1] | COUNT [0.1] | IF_ELSE [0.1]
                        
EXPR_NO_IE -> EXPR_NO_IE '+' TERM [0.25] | EXPR_NO_IE '-' TERM [0.25] | TERM [0.25] |  MIN_MAX [0.15] | COUNT [0.1] 
                        
TERM -> VAR [0.4] | INTEGER [0.2] | '(' EXPR ')' [0.4]

MIN_MAX -> 'min(' EXPR ',' EXPR ')' [0.5] | 'max(' EXPR ',' EXPR ')' [0.5]

COUNT -> 'arr.count(' INTEGER ')' [1.0]

BOOLEAN -> 'not' BOOLEAN [0.10] | BOOLEAN 'and' BOOLEAN [0.10] | BOOLEAN 'or' BOOLEAN [0.10] | EXPR_NO_IE '==' EXPR_NO_IE [0.10] | EXPR_NO_IE '!=' EXPR_NO_IE [0.10] | EXPR_NO_IE '<' EXPR_NO_IE [0.10] | EXPR_NO_IE '<=' EXPR_NO_IE [0.10] | EXPR_NO_IE '>' EXPR_NO_IE [0.10] | EXPR_NO_IE '>=' EXPR_NO_IE [0.10] | 'True' [0.05] | 'False' [0.05]

IF_ELSE -> EXPR_NO_IE 'if' BOOLEAN 'else' EXPR [1.0]

INTEGER -> '0' [0.1] | '1' [0.1] | '2' [0.1] | '3' [0.1] | '4' [0.1] | '5' [0.1] | '6' [0.1] | '7' [0.1] | '8' [0.1] | '9' [0.1]

VAR -> 'x' [0.4] | 'y' [0.3] | 'z' [0.3]"""


FOM_GRAMMAR = """
S -> FUNC [1.0]

FUNC -> 'lambda' 'x, y, z, arr' ':' EXPR [1.0]

EXPR -> EXPR '+' TERM [0.15] | EXPR '-' TERM [0.15] | EXPR '*' TERM [0.20] | TERM [0.2] |  MIN_MAX [0.1] | COUNT [0.1] | IF_ELSE [0.1]
                        
EXPR_NO_IE -> EXPR_NO_IE '+' TERM [0.15] | EXPR_NO_IE '-' TERM [0.15] | EXPR_NO_IE '*' TERM [0.20] | TERM [0.25] |  MIN_MAX [0.15] | COUNT [0.1] 
                        
TERM -> VAR [0.4] | INTEGER [0.2] | '(' EXPR ')' [0.4]

MIN_MAX -> 'min(' EXPR ',' EXPR ')' [0.5] | 'max(' EXPR ',' EXPR ')' [0.5]

COUNT -> 'arr.count(' INTEGER ')' [1.0]

BOOLEAN -> 'not' BOOLEAN [0.10] | BOOLEAN 'and' BOOLEAN [0.10] | BOOLEAN 'or' BOOLEAN [0.10] | EXPR_NO_IE '==' EXPR_NO_IE [0.10] | EXPR_NO_IE '!=' EXPR_NO_IE [0.10] | EXPR_NO_IE '<' EXPR_NO_IE [0.10] | EXPR_NO_IE '<=' EXPR_NO_IE [0.10] | EXPR_NO_IE '>' EXPR_NO_IE [0.10] | EXPR_NO_IE '>=' EXPR_NO_IE [0.10] | 'True' [0.05] | 'False' [0.05]

IF_ELSE -> EXPR_NO_IE 'if' BOOLEAN 'else' EXPR [1.0]

INTEGER -> '0' [0.1] | '1' [0.1] | '2' [0.1] | '3' [0.1] | '4' [0.1] | '5' [0.1] | '6' [0.1] | '7' [0.1] | '8' [0.1] | '9' [0.1]

VAR -> 'x' [0.4] | 'y' [0.3] | 'z' [0.3]"""


def generate_dyck(num_symbols, max_length=510):
    """Generates a Dyck sequence with specified number of symbols.

    Args:
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        max_length: The maximum length of the generated sequence.

    Returns:
        A list representing the Dyck sequence, or None if generation fails
        (e.g., due to exceeding max_length prematurely).
    """

    result = []
    stack = []

    while len(result) < max_length:
        if not stack:  # If the stack is empty, we must open a new bracket
            opening_symbol = np.random.randint(0, num_symbols)
            result.append(opening_symbol)
            stack.append(opening_symbol)
        else:
            if random.random() < 0.5:  # Decide whether to open or close
                if len(result) >= max_length - 1:
                    closing_symbol = stack.pop() + num_symbols
                    result.append(closing_symbol)
                    if not stack:
                        break
                    continue
                opening_symbol = np.random.randint(0, num_symbols)
                result.append(opening_symbol)
                stack.append(opening_symbol)

            else:
                if stack:
                    closing_symbol = stack.pop() + num_symbols
                    result.append(closing_symbol)
                else:
                    break  # prevent infinite loop

    if stack:  # if stack is not empty, then we did not finish correctly. Return none
        return None
    return result


def generate_dyck_txt_file(file_dir, num_symbols=30, n=100000, target_length=2048):
    """Generates a text file containing Dyck sequences.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        n: The number of sequences to generate.
    """
    import os

    os.makedirs(file_dir, exist_ok=True)
    with open(f"{file_dir}/dyck_sequences_{num_symbols}.txt", "w") as f:
        for i in trange(n):
            result = []
            while len(result) < target_length:
                new_seq = generate_dyck(num_symbols)
                if new_seq is None:
                    continue
                result.extend(new_seq)

            dyck_str = " ".join(
                [str(x) for x in result[:target_length]]
            )  # truncate to target length
            f.write(f"{dyck_str}\n")


def generate_repeated_token_sequences(
    file_dir, num_symbols=30, k=10, n=100000, target_length=2048
):
    """Generates n strings of target_length, each composed of repetitions
    of a *single* k-length random token sequence.

    Args:
        num_symbols: The number of distinct symbols.
        k: The length of the repeating token sequence.
        n: The number of sequences to generate.
        target_length: The desired length of each generated string.
        file_dir: The directory to save the generated sequences.
    """

    os.makedirs(file_dir, exist_ok=True)
    with open(f"{file_dir}/repeated_token_sequences_{num_symbols}_{k}.txt", "w") as f:
        for _ in trange(n):
            # Generate the k-length token sequence *once* per output sequence
            token_sequence = np.random.randint(0, num_symbols, size=k).tolist()
            sequence = []
            while len(sequence) < target_length:
                sequence.extend(token_sequence)  # Repeat the same sequence
            sequence = sequence[:target_length]  # Truncate to the target length
            sequence_str = " ".join(map(str, sequence))
            f.write(f"{sequence_str}\n")


def generate_dyck_cross_serial(num_symbols, max_length=510, p=0.5):
    """Generates a Dyck sequence with cross-serial dependencies,
    meaning parentheses only need to be balanced, not necessarily nested.

    Args:
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        max_length: The maximum length of the generated sequence.

    Returns:
        A list representing the Dyck sequence with cross-serial dependencies,
        or None if generation fails (e.g., exceeding max_length).
    """

    result = []
    counts = [0] * num_symbols  # Keep track of open counts for each symbol

    while len(result) < max_length:
        if all(c == 0 for c in counts):
            # If all counts are zero, we must open a new bracket
            opening_symbol = np.random.randint(0, num_symbols)
            result.append(opening_symbol)
            counts[opening_symbol] += 1
        else:
            if random.random() < p:
                if len(result) >= max_length - 1:
                    for symbol in range(num_symbols):
                        while counts[symbol] > 0:
                            result.append(symbol + num_symbols)
                            counts[symbol] -= 1
                    if not all(c == 0 for c in counts):
                        return None
                    break
                # Decide to open a new bracket
                opening_symbol = np.random.randint(0, num_symbols)
                result.append(opening_symbol)
                counts[opening_symbol] += 1
            else:
                # Decide to close an existing bracket

                open_symbols = [i for i, c in enumerate(counts) if c > 0]

                if not open_symbols:
                    # prevent infinite loop
                    # force open a new bracket
                    if len(result) >= max_length - 1:
                        for symbol in range(num_symbols):
                            while counts[symbol] > 0:
                                result.append(symbol + num_symbols)
                                counts[symbol] -= 1
                        if not all(c == 0 for c in counts):
                            return None
                        break
                    opening_symbol = np.random.randint(0, num_symbols)
                    result.append(opening_symbol)
                    counts[opening_symbol] += 1

                else:
                    closing_symbol = np.random.choice(open_symbols)
                    result.append(closing_symbol + num_symbols)
                    counts[closing_symbol] -= 1

    if not all(c == 0 for c in counts):
        return None

    return result


def generate_dyck_txt_file_cross_serial(
    file_dir, num_symbols=30, n=100000, target_length=2048, p=0.5
):
    """Generates a text file containing Dyck sequences with cross-serial dependencies.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        n: The number of sequences to generate.
        target_length: desired sequence length.
    """
    import os

    os.makedirs(file_dir, exist_ok=True)
    print(num_symbols, p)
    # if exists, return
    if os.path.exists(f"{file_dir}/dyck_sequences_cross_serial_{num_symbols}_{p}.txt"):
        print("File already exists. Skipping generation.")
        return
    with open(
        f"{file_dir}/dyck_sequences_cross_serial_{num_symbols}_{p}.txt", "w"
    ) as f:
        for i in trange(n):
            result = []
            while len(result) < target_length:
                new_seq = generate_dyck_cross_serial(
                    num_symbols, max_length=target_length - len(result), p=p
                )
                if new_seq is None:
                    continue
                result.extend(new_seq)

            dyck_str = " ".join(
                [str(x) for x in result[:target_length]]
            )  # truncate to target length
            f.write(f"{dyck_str}\n")


def generate_grammar(
    n: int = 100,
    depth: int = 1,
    outdir: str = "../data/generated_grammar.txt",
    seed: int = 0,
    gen_string_name: str = "CRASP",
):
    if seed is not None:
        random.seed(seed)

    # Define the probabilistic grammar
    gen_string = globals()[f"{gen_string_name}_GRAMMAR"]
    pgrammar = PCFG.fromstring(gen_string)

    def generate_random_derivation(symbol, current_depth):
        if current_depth == 0:
            # If depth is 0, only return terminal symbols
            if isinstance(symbol, str):  # Terminal symbols
                return [symbol]
            else:
                return [
                    random.choice(
                        [
                            "x",
                            "y",
                            "z",
                        ]
                    )
                ]

        if isinstance(symbol, str):
            # Terminal symbol, return it directly
            return [symbol]

        # Find all production rules for the given non-terminal symbol
        productions = pgrammar.productions(lhs=symbol)

        # Randomly choose a production rule based on probability
        production = random.choices(
            productions, weights=[prod.prob() for prod in productions]
        )[0]

        # Recursively expand all symbols in the production
        result = []
        for part in production.rhs():
            result.extend(generate_random_derivation(part, current_depth - 1))

        return result

    import os

    os.makedirs(outdir, exist_ok=True)

    for i in range(n):
        sentence = generate_random_derivation(pgrammar.start(), depth)
        # print(' '.join(sentence))
        # write to file
        with open(os.path.join(outdir, f"{i}.txt"), "w") as f:
            f.write(" ".join(sentence) + "\n")


def eval_with_intermediate_steps(expression, x, y, z, arr):
    # Parse the expression into an AST
    tree = ast.parse(expression, mode="eval")
    cot_buffer = []

    # Define a visitor class to traverse the AST and print intermediate steps
    class Evaluator(ast.NodeVisitor):
        def visit_Constant(self, node):
            # Handle constant values directly
            return node.value

        def visit_BinOp(self, node):
            # Before evaluating a binary operation, print the operation
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = type(node.op).__name__
            cot_buffer.append(f"BinOp: {left} {op} {right}.")
            return self.evaluate_op(left, right, op)

        def visit_IfExp(self, node):
            # Print the evaluation of a conditional expression
            condition = self.visit(node.test)
            cot_buffer.append(
                f"IfExp: {condition} ? {self.visit(node.body)} : {self.visit(node.orelse)}."
            )
            return condition and self.visit(node.body) or self.visit(node.orelse)

        def visit_Call(self, node):
            # Handle function calls and method calls
            func = self.visit(node.func)
            args = [self.visit(arg) for arg in node.args]

            if isinstance(node.func, ast.Attribute):
                # Method call like arr.count(5)
                obj = self.visit(node.func.value)
                method_name = node.func.attr
                cot_buffer.append(f"Call: {method_name} on {obj} with args {args}.")
                return self.evaluate_method_call(obj, method_name, args)
            else:
                # Regular function call
                cot_buffer.append(f"Call: {func} with args {args}.")
                return self.evaluate_function_call(func, args)

        def visit_Compare(self, node):
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            op = type(node.ops[0]).__name__  # The comparison operator (e.g., 'Lt')
            cot_buffer.append(f"Compare: {left} {op} {right}.")
            return self.evaluate_comparison(left, right, op)

        def visit_Name(self, node):
            # Evaluate variables
            if node.id == "x":
                return x
            elif node.id == "y":
                return y
            elif node.id == "z":
                return z
            elif node.id == "arr":
                return arr
            elif node.id in ["min", "max"]:
                return node.id
            else:
                raise ValueError(f"Unknown variable {node.id}")

        def visit_List(self, node):
            # Handle list literals
            return [self.visit(elem) for elem in node.elts]

        def visit_BoolOp(self, node):
            # Handle boolean operations (and, or)
            values = [self.visit(value) for value in node.values]
            op = type(node.op).__name__
            cot_buffer.append(f"BoolOp: {op} with values {values}.")
            return self.evaluate_bool_op(values, op)

        def visit_UnaryOp(self, node):
            # Handle unary operations (not)
            operand = self.visit(node.operand)
            op = type(node.op).__name__
            cot_buffer.append(f"UnaryOp: {op} {operand}.")
            return self.evaluate_unary_op(operand, op)

        def visit(self, node):
            method_name = "visit_" + type(node).__name__
            visitor = getattr(self, method_name, self.generic_visit)
            return visitor(node)

        def evaluate_op(self, left, right, op_type):
            # Handle basic operations
            ops = {
                "Add": operator.add,
                "Sub": operator.sub,
                "Mult": operator.mul,
            }
            return ops[op_type](left, right)

        def evaluate_method_call(self, obj, method_name, args):
            # Handle method calls
            if method_name == "count":
                return obj.count(args[0])
            else:
                raise ValueError(f"Unsupported method: {method_name}")

        def evaluate_function_call(self, func_name, args):
            # Handle function calls
            if func_name == "min":
                return min(args)
            elif func_name == "max":
                return max(args)
            elif func_name == "count":
                # Assumes count is implemented as counting occurrences in a list
                if len(args) != 2:
                    raise ValueError("count() requires two arguments: list and value")
                return args[0].count(args[1])
            else:
                raise ValueError(f"Unsupported function: {func_name}")

        def evaluate_comparison(self, left, right, op_type):
            # Handle comparisons
            ops = {
                "Lt": operator.lt,
                "LtE": operator.le,
                "Gt": operator.gt,
                "GtE": operator.ge,
                "Eq": operator.eq,
                "NotEq": operator.ne,
            }
            return ops[op_type](left, right)

        def evaluate_bool_op(self, values, op_type):
            # Handle boolean operations
            if op_type == "And":
                return all(values)
            elif op_type == "Or":
                return any(values)
            else:
                raise ValueError(f"Unsupported boolean operation: {op_type}")

        def evaluate_unary_op(self, operand, op_type):
            # Handle unary operations
            if op_type == "Not":
                return not operand
            else:
                raise ValueError(f"Unsupported unary operation: {op_type}")

    evaluator = Evaluator()
    ret = evaluator.visit(tree.body)
    cot_buffer.append(f"Result: {int(ret)}.")
    cot = " ".join(cot_buffer)

    return cot


def eval_txt_file(file_dir, n: int = 256):
    file_paths = glob.glob(f"{file_dir}/*.txt")
    for file_path in tqdm(file_paths):
        with open(file_path, "r") as f:
            first_line = f.readline()
            lambda_expr = first_line.strip()

        expr = lambda_expr.split(":")[-1].strip()

        with open(file_path, "w") as f:
            f.write(first_line)  # Write the first line back to the file
            for _ in range(n):
                x, y, z = (
                    random.randint(0, 9),
                    random.randint(0, 9),
                    random.randint(0, 9),
                )
                arr_len = random.randint(4, 8)
                arr = [random.randint(0, 9) for _ in range(arr_len)]
                cot = eval_with_intermediate_steps(expr, x, y, z, arr)
                ret = f"(x={x}, y={y}, z={z}, arr={arr}) -> {cot}\n"

                f.write(ret)


def main():
    fire.Fire(
        {
            "generate": generate_grammar,
            "eval": eval_txt_file,
            "generate_dyck": generate_dyck_txt_file,
            "generate_rep": generate_repeated_token_sequences,
            "generate_cross": generate_dyck_txt_file_cross_serial,
        }
    )


if __name__ == "__main__":
    main()
