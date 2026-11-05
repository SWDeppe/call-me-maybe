# from llm_sdk import Small_LLM_Model
# import torch
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import TypeAdapter

from .Function import Function, FunctionDefinition, PromptDefinition

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def select_function(
    prompt: str, functions: list[Function], sdk: Small_LLM_Model
) -> int:
    """ " select one of the valide function"""
    # prepare a prompte
    new_prompt = "Available functions:\n"
    new_prompt += "    0. No function matches\n"
    for i, func in enumerate(functions):
        new_prompt += f"    {i + 1}. {func.__repr__()}\n"
    new_prompt += f"Request: {prompt}"
    new_prompt += "\n\nFunction Name: "

    functions_token: list[list[int]] = [
        sdk.encode(func.name).squeeze(0).tolist() for func in functions
    ]
    i = 0
    function_name = ""
    # TODO: give a token of no choice that compeet with the function's names
    while True:
        logits = sdk.get_logits_from_input_ids(
            sdk.encode(new_prompt).squeeze(0).tolist()
        )
        functions_tokens_scores = [
            logits[tokens[i]] for tokens in functions_token if len(tokens) > i
        ]
        choice = np.argmax(functions_tokens_scores)
        function_name += sdk.decode([functions_token[choice][i]])
        new_prompt += sdk.decode([functions_token[choice][i]])
        # define the remaning possible functions to choose
        token = functions_token[choice][i]
        functions_token = [tokens for tokens
                           in functions_token if tokens[i] == token]
        if len(functions_token) == 1:
            function_name = sdk.decode(functions_token[0])
            break
        i += 1

    for i, func in enumerate(functions):
        if func.name == function_name:
            return i
    return -1


def get_possible_argumets(prompt: str) -> dict[str, str]:

    strings: list[str] = [
        string for i, string in enumerate(prompt.split('"')) if i % 2 == 1
    ]
    rest: list[str] = [
        string for i, string in enumerate(prompt.split('"')) if i % 2 == 0
    ]

    arguments: dict[str, str] = {}
    for part in rest:
        numbers = [
            float(value)
            for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])",
                                    prompt)
        ]

        for number in numbers:
            if str(number) not in arguments:
                if number >= 0:
                    arguments["+" + str(number)] = "number"
                else:
                    arguments[str(number)] = "number"
        if part != rest[-1]:
            new_str = strings.pop(0)
            arguments[new_str] = "string"

    return arguments


def prompt_number(
    prompt: str, delim: str, nth_arg: float, sdk: Small_LLM_Model
) -> float:
    """takes a prompt and generate a number till delime
    (delime should be one single token !)"""
    if len(sdk.encode(delim).squeeze(0).tolist()) != 1:
        raise ValueError("delime is more than one token !")
    DIGITS = "0123456789"
    SIGNE = "-+"
    i = 0
    new_prompt = prompt
    number: str = ""
    while True:
        logits = sdk.get_logits_from_input_ids(
            sdk.encode(new_prompt).squeeze(0).tolist()
        )
        candidates = (
            SIGNE
            if i == 0
            else DIGITS + ("" if i == 1 else ".") + ("" if i == 0 else delim)
        )
        # print(f"for i {i}:", candidates)
        c_ids: list[list[int]] = [
            sdk.encode(candidate).squeeze(0).tolist()
            for candidate in candidates
        ]
        if next((ids for ids in c_ids if len(ids) == 1), None) is None:
            raise IndexError("all the candidates are not 1 token...")
        scores: list[float] = [logits[id[0]] for id in c_ids]
        if i == 0:
            print(scores)
            # scores[0] += 1
            scores[0] *= 1 + nth_arg * 0.25

        choice = np.argmax(scores)
        if candidates[choice] == delim:
            return float(number)
        number += candidates[choice]
        new_prompt += candidates[choice]
        i += 1


def prompt_string(prompt: str, sdk: Small_LLM_Model) -> str:
    result = "'"
    while True:
        logits = sdk.get_logits_from_input_ids(
            sdk.encode(prompt + result).squeeze(0).tolist()
        )
        i = np.argmax(logits)
        val = sdk.decode([i])
        if "\n" in val:
            for char in val:
                if char == "'" or char == "\n":
                    break
                result += char
            return result[1:]
        result += sdk.decode([i])


def get_arguments(
    prompt: str, function: Function, sdk: Small_LLM_Model
) -> dict[str, str | float]:
    # "parse the prompt to find arguments to hint"
    args = get_possible_argumets(prompt)

    # make the prompt
    new_prompt = f"Function:{function}\n\nRequest: {prompt}\n\n"
    new_prompt += "Preserve the user's requested meaning !\n" \
                  f"Make shure to handle {function.name}'s spescial cases" \
                  "\nReturn only the parameter value."
    new_prompt += "Dont foret to repeat the signes for number type\n" \
                  "Parameter choice format:\n" \
                  "name(type)=value\nname(type)=value\n"
    new_prompt += (
        f"\nParameters choice for request {prompt}" +
        " with the function {function.name}:\n"
    )

    if len(args) != 0:
        new_prompt += f"Hint: {args}\n"

    # generate argument by argument
    responses: dict[str, str | float] = {}
    neg_multi = 0.0
    for param_name, param_value in function.parameters.items():
        new_prompt += f"\t{param_name}({param_value})="
        val: float | str
        if param_value == "number":
            val = prompt_number(new_prompt, "\n", neg_multi, sdk)
            responses[param_name] = val
            if val < 0:
                neg_multi += 2.5
            neg_multi += 1.0
        elif param_value == "string":
            val = prompt_string(new_prompt, sdk)
            responses[param_name] = val
        else:
            raise (ValueError("type can juste be string or number"))
        if str(val) != "":
            new_prompt += str(val) + (
                f"\n\tnew arg for '{prompt}' with the -\n"
                if param_value == "number"
                else "\n"
            )

    return responses


def load_json(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except OSError as exc:
        raise ValueError(f"{path}: impossible to read file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: JSON invalide: {exc.msg}"
        ) from exc


def parse(
    func_definition_path: str,
    prompts_path: str,
) -> tuple[list[FunctionDefinition], list[PromptDefinition]]:
    functions = TypeAdapter(list[FunctionDefinition]).validate_python(
        load_json(func_definition_path)
    )
    prompts = TypeAdapter(list[PromptDefinition]).validate_python(
        load_json(prompts_path)
    )
    return functions, prompts


def convert_to_func(definition: FunctionDefinition) -> Function:
    parameters = {
        name: value.type for name, value in definition.parameters.items()
    }
    return Function(
        definition.name,
        definition.description,
        definition.returns.type,
        parameters,
    )


def call_me_maybe() -> None:
    from llm_sdk import Small_LLM_Model

    sdk = Small_LLM_Model()
    here = Path().absolute()
    functions, prompts = parse(
        str(here) + "/data/input/functions_definition.json",
        str(here) + "/data/input/function_calling_tests.json",
    )
    my_functions: list[Function] = []
    for func in functions:
        my_functions.append(convert_to_func(func))
    func_selections: list[Function] = []
    args_selections: list[dict[str, str | float]] = []
    prompts_selections: list[str] = []
    for prompt in prompts:
        func_i = select_function(prompt.prompt, my_functions, sdk)
        args = get_arguments(prompt.prompt, my_functions[func_i], sdk)
        print(f"{prompt.prompt}: {my_functions[func_i].name}(", end="")
        print(*args.values(), sep=", ", end="")
        print(")")
        prompts_selections.append(prompt.prompt)
        func_selections.append(my_functions[func_i])
        args_selections.append(args)
        print()
    output(prompts_selections, func_selections, args_selections)


def output(
    prompts: list[str],
    functions: list[Function],
    args: list[dict[str, str | float]],
    output_path: str = "data/output/function_calling_results.json",
) -> None:
    if len(prompts) != len(functions) or len(prompts) != len(args):
        raise ValueError(
            "Output's len of functions,args, prompts dose't corresponds !!"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    responses: list[dict[str, str | dict[str, str | float]]] = []
    for i in range(len(prompts)):
        as_dict: dict[str, str | dict[str, str | float]] = {
            "prompt": prompts[i],
            "function": functions[i].name,
            "parameters": args[i],
        }
        responses.append(as_dict)
    with open(output_path, "w+") as f:
        json.dump(responses, f, indent=4)


def main() -> None:
    call_me_maybe()
