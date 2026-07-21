from experiments_2d.config import load_config
from experiments_2d.constants import DIFFICULTIES, ENV_TO_CATEGORY
from experiments_2d.data import smoke_subset
from experiments_2d.prompting import all_type_tools


def test_config_freezes_single_gpu_qwen_protocol() -> None:
    config = load_config("experiments_2d/configs/qwen3_4b_instruct_2507.yaml")
    assert config.model.num_hidden_layers == 36
    assert config.model.hidden_size == 2560
    assert config.generation.tensor_parallel_size == 1
    assert config.analysis.onset_peak_fraction == 0.95


def test_smoke_subset_has_one_task_per_environment_difficulty_cell() -> None:
    tasks = []
    task_id = 1
    for environment in ENV_TO_CATEGORY:
        for difficulty in DIFFICULTIES:
            for _ in range(2):
                tasks.append(
                    {
                        "id": task_id,
                        "difficulty": difficulty,
                        "environments": [{"name": environment}],
                    }
                )
                task_id += 1
    selected = smoke_subset(list(reversed(tasks)))
    cells = {
        (task["environments"][0]["name"], task["difficulty"])
        for task in selected
    }
    assert len(selected) == 45
    assert len(cells) == 45


def test_p_all_menu_is_fixed_three_type_tools() -> None:
    tools = all_type_tools()
    names = [tool["function"]["name"] for tool in tools]
    assert names == [
        "type_a_computational_scale",
        "type_b_knowledge_boundary",
        "type_c_reliable_execution",
    ]

