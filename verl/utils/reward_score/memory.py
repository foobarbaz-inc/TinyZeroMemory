import random
import re


def extract_output(solution_str):
    """Extract the output from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split("\n")[-1]

    output_pattern = r"<output>(.*?)</output>"
    match = re.finditer(output_pattern, solution_str)
    matches = list(match)
    if matches:
        final_output = matches[-1].group(1).strip()
    else:
        final_output = None
    return final_output


def extract_tool_params_from_output(output_str):
    """Extract kwargs from a function call string like 'READ_INFO(full_name="David Kraft", column_name="age")'.

    Returns:
        tuple: (full_name, column_name) if both parameters are found
        None: if the string doesn't match the expected format
    """
    # Match pattern for kwargs in function call
    pattern = r'READ_INFO\(full_name="([^"]*)",\s*column_name="([^"]*)"\)'
    match = re.match(pattern, output_str)

    if match:
        return (match.group(1), match.group(2))
    return None


def compute_score(
    solution_str,
    ground_truth,
    format_score_output=0.1,
    format_score_tool_params=0.2,
    score=1.0,
):
    target_tool_call = ground_truth["tool_call_1"]

    do_print = random.randint(1, 64) == 1

    output = extract_output(solution_str)

    if do_print:
        print("--------------------------------")
        print(f"Target: {target_tool_call}")
        print(f"Extracted output: {output}")
        print(f"Solution string: {solution_str}")

    if output is None:
        if do_print:
            print("No output found")
        return 0

    tool_params = extract_tool_params_from_output(output)
    if tool_params is None:
        if do_print:
            print("No tool params found")
        return format_score_output

    tool_call_full_name, tool_call_column_name = tool_params
    target_full_name, target_column_name = target_tool_call

    if (
        tool_call_full_name != target_full_name
        or tool_call_column_name != target_column_name
    ):
        if do_print:
            print("Tool call params do not match target")
        return format_score_tool_params

    return score
