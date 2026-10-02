"""Prompt templates with the task description, optional demonstrations and then the query."""


def labeled(input_label: str, output_sep: str, zero_shot_label: str):
    """Template for 9 tasks, in which each demonstration is `<label><input><sep><answer>`."""

    def format_prompt(sample, examples=None):
        if not examples:
            return f"{sample.task_description}\n\n{zero_shot_label}{sample.input}"
        parts = [sample.task_description, "\nHere are some examples:\n"]
        for ex in examples:
            parts.append(f"{input_label}{ex.input}{output_sep}{ex.gold_answer}\n\n---\n")
        parts.append(f"{input_label}{sample.input}")
        return "\n".join(parts)

    return format_prompt


def acebench(sample, examples=None):
    if not examples:
        return f"{sample.task_description}\n\n{sample.input}"
    parts = [sample.task_description, "\n\nHere are some examples:\n"]
    for ex in examples:
        parts += [f"\n{ex.input}", f"\n\nResponse: {ex.gold_answer}", "\n\n---\n"]
    parts.append(f"\n{sample.input}")
    return "".join(parts)


def ifbench(sample, examples=None):
    if not examples:
        return f"{sample.task_description}\n\n{sample.input}"
    parts = [sample.task_description, "\n\nHere are some examples:\n"]
    for ex in examples:
        parts += [f"{ex.input}\n\nResponse: {ex.gold_answer}", "\n\n---\n\n"]
    parts.append(sample.input)
    return "".join(parts)


def sifo(sample, examples=None):
    if not examples:
        return f"{sample.task_description}\n\n{sample.input}"
    parts = [sample.task_description, "\nHere are some examples:"]
    for ex in examples:
        parts.append(f"\n{ex.input}\n\nResponse: {ex.gold_answer}\n\n---")
    parts.append(f"\n{sample.input}")
    return "\n".join(parts)


def docstring(sample, examples=None):
    if not examples:
        return f"{sample.task_description}\n\n{sample.input}"
    # The description's "Input:" and "Output:" lines would look like demonstration markers.
    desc = "\n".join(
        line for line in sample.task_description.split("\n")
        if not line.startswith("Input: ") and not line.startswith("Output: ")
    ).rstrip()
    parts = [desc, "\n\nHere are some examples:\n"]
    for ex in examples:
        parts += [f"{ex.input}\n\nDocstring: {ex.gold_answer}", "\n\n---\n\n"]
    parts.append(f"Now generate the docstring for the following function:\n\n{sample.input}")
    return "".join(parts)
