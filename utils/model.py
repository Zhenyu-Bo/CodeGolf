import hydra
import random

from omegaconf import OmegaConf, DictConfig, ListConfig
from qwen_agent.llm import get_chat_model, BaseChatModel


def get_qwen_agent_model(model_config: dict | DictConfig) -> BaseChatModel:
    if isinstance(model_config, (DictConfig, ListConfig)):
        model_config = OmegaConf.to_container(model_config)
    if "model_server" in model_config and isinstance(model_config["model_server"], list):
        model_config["model_server"] = random.choice(model_config["model_server"])
    if "api_keys" in model_config:
        api_keys = model_config["api_keys"].split(",")
        model_config["api_key"] = random.choice(api_keys)
    return get_chat_model(model_config)


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig):
        OmegaConf.resolve(cfg)
        # print(f"Config: {cfg.model}")
        model = get_qwen_agent_model(cfg.model)
        responses = model.chat([{"role": "user", "content": "How AI works?"}], stream=False)
        print(responses)
        # breakpoint()

    main()
