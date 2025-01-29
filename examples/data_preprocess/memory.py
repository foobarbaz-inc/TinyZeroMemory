import argparse
import csv
import datetime
import os
import random
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs

# dict from question type to question prompt template
QUESTION_TEMPLATES = {
    # single output
    "WHO_IS_YOUNGER": "Who is younger, {name_a} or {name_b}?",
    "SAME_BIRTHDAY": "Do {name_a} and {name_b} have the same birthday?",
    "SAME_CITY": "Do {name_a} and {name_b} live in the same city?",
    "AGE_DIFFERENCE": "What is the age difference between {name_a} and {name_b}?",
    "SAME_JOB": "Work {name_a} and {name_b} the same job?",
    # two outputs separated by a comma
    # "How old are {name_a} and {name_b} (output in order in years, separated by a comma)?",
}


def _load_list_file(filename: str) -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "r") as f:
        return [line.strip().capitalize() for line in f.readlines()]


def _read_table(file_path: str) -> List[dict]:
    people_data = []
    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people_data.append(row)
    return people_data


def generate_dataset_on_huggingface(
    people_data: List[dict], num_samples: int, huggingface_path: str
) -> pd.DataFrame:
    samples = []

    for _ in tqdm(range(num_samples)):
        question_type = random.choice(list(QUESTION_TEMPLATES.keys()))
        person_a, person_b = random.sample(people_data, 2)

        sample = {
            "question_type": question_type,
            "name_a": person_a["full_name"],
            "name_b": person_b["full_name"],
            "tool_call_1": (person_a["full_name"], "age"),
            "tool_call_1_result": None,
            "tool_call_2": (person_b["full_name"], "age"),
            "tool_call_2_result": None,
            "answer": None,
        }

        if question_type == "WHO_IS_YOUNGER":
            sample.update(
                {
                    "tool_call_1": (person_a["full_name"], "age"),
                    "tool_call_1_result": person_a["age"],
                    "tool_call_2": (person_b["full_name"], "age"),
                    "tool_call_2_result": person_b["age"],
                    "answer": person_a["full_name"]
                    if person_a["age"] < person_b["age"]
                    else person_b["full_name"],
                }
            )
        elif question_type == "SAME_BIRTHDAY":
            sample.update(
                {
                    "tool_call_1": (person_a["full_name"], "birthday"),
                    "tool_call_1_result": person_a["birthday"],
                    "tool_call_2": (person_b["full_name"], "birthday"),
                    "tool_call_2_result": person_b["birthday"],
                    "answer": "YES"
                    if person_a["birthday"] == person_b["birthday"]
                    else "NO",
                }
            )
        elif question_type == "SAME_CITY":
            sample.update(
                {
                    "tool_call_1": (person_a["full_name"], "city"),
                    "tool_call_1_result": person_a["city"],
                    "tool_call_2": (person_b["full_name"], "city"),
                    "tool_call_2_result": person_b["city"],
                    "answer": "YES" if person_a["city"] == person_b["city"] else "NO",
                }
            )
        elif question_type == "AGE_DIFFERENCE":
            sample.update(
                {
                    "tool_call_1": (person_a["full_name"], "age"),
                    "tool_call_1_result": person_a["age"],
                    "tool_call_2": (person_b["full_name"], "age"),
                    "tool_call_2_result": person_b["age"],
                    "answer": str(abs(int(person_a["age"]) - int(person_b["age"]))),
                }
            )
        elif question_type == "SAME_JOB":
            sample.update(
                {
                    "tool_call_1": (person_a["full_name"], "occupation"),
                    "tool_call_1_result": person_a["occupation"],
                    "tool_call_2": (person_b["full_name"], "occupation"),
                    "tool_call_2_result": person_b["occupation"],
                    "answer": "YES"
                    if person_a["occupation"] == person_b["occupation"]
                    else "NO",
                }
            )

        samples.append(sample)

    df = pd.DataFrame(samples)
    df.to_parquet(f"hf://datasets/{huggingface_path}")

    return pd.DataFrame(samples)


def generate_table(num_people: int, file_path: str):
    first_names = _load_list_file("random_lists/first_names.txt")
    last_names = _load_list_file("random_lists/last_names.txt")
    cities = _load_list_file("random_lists/places.txt")
    occupations = _load_list_file("random_lists/occupations.txt")

    base_date = datetime.datetime(2024, 1, 1)
    all_dates = [
        (base_date + datetime.timedelta(days=x)).strftime("%B %-d") for x in range(366)
    ]

    # Generate records and store them in a set to ensure uniqueness
    unique_records = set()
    pbar = tqdm(total=num_people)

    while len(unique_records) < num_people:
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        city = random.choice(cities)
        occupation = random.choice(occupations)
        birthday = random.choice(all_dates)
        age = random.randint(18, 80)

        # Create a tuple of the record (immutable for set storage)
        record = (full_name, city, occupation, birthday, age)

        if record not in unique_records:
            unique_records.add(record)
            pbar.update(1)

    pbar.close()

    # Write unique records to file
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["full_name", "city", "occupation", "birthday", "age"])
        writer.writerows(unique_records)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Generate random people data and create a dataset"
#     )
#     parser.add_argument(
#         "--num-samples",
#         type=int,
#         default=1000,
#         help="Number of samples to generate for the dataset (default: 1000)",
#     )
#     parser.add_argument(
#         "--huggingface-path",
#         type=str,
#         required=True,
#         help="Path to save the Hugging Face dataset",
#     )

#     args = parser.parse_args()

#     people_data = _read_table("people_data.csv")
#     dataset = generate_dataset_on_huggingface(
#         people_data, args.num_samples, args.huggingface_path
#     )


def make_prefix(dp, template_type):
    if template_type == "base":
        formatted_question = QUESTION_TEMPLATES[dp["question_type"]].format(
            name_a=dp["name_a"], name_b=dp["name_b"]
        )
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it by using the appropriate tools. The assistant first thinks about the reasoning process and then outputs the next step in the process which is either a tool call or the answer.

User: Using the available tools, answer the following question: {formatted_question}. You can use the <output>READ_INFO(full_name: str, column_name: str <city, occupation, birthday, age>)</output> tool in your output to lookup information (example valid tool call with values filled in to get a person's age: <output>READ_INFO(full_name="Michael Schulz", column_name="age")</output>). Show your work in <think> </think> tags and return the final output, either a tool call or your answer, in <output></output> tags.
Assistant: Let me solve this step by step.
<think>"""

    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/memory")
    parser.add_argument("--hdfs_dir", default=None)
    # parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--train_size", type=int, default=95000)
    parser.add_argument("--test_size", type=int, default=4000)
    parser.add_argument("--template_type", type=str, default="base")

    args = parser.parse_args()

    data_source = "memory"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset("Luca-nc/memory", split="train")

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "tool_call_1": example["tool_call_1"],
                "tool_call_1_result": example["tool_call_1_result"],
                "tool_call_2": example["tool_call_2"],
                "tool_call_2_result": example["tool_call_2_result"],
                "answer": example["answer"],
            }
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
