from typing import Callable, Dict, List

from models.basenet import BaseNet
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN


class NetFactory:
    @classmethod
    def get_net(cls, class_name: str, net_id: int, params: List[str]) -> BaseNet:
        params = cls.parse_params(params)
        builder = cls.get_net_builder(class_name)
        return builder(net_id, *params)

    @staticmethod
    def get_net_builder(class_name: str) -> Callable:
        builders = {
            "lenetlike": LeNetLike,
            "vanilla_cnn": VanillaCNN,
            "resnet_fusedbn": ResNetFusedBN,
        }
        return builders[class_name]

    @classmethod
    def converters(cls) -> Dict[str, Callable]:
        return {"f": float, "d": int, "l": cls.list_converter, "s": str}

    @classmethod
    def parse_params(cls, params: List[str]) -> List:
        result = []
        converters = cls.converters()

        for p in params:
            if len(p) > 1:
                t, v = p[0], p[1:]
                result.append(converters[t](v))

        return result

    @classmethod
    def list_converter(cls, s: str) -> List:
        if s[0] != "[" or s[-1] != "]":
            raise ValueError

        s = s[1:-1]

        if "*" in s:
            elements = s.split("*")
        else:
            elements = s.split(",")

        return cls.parse_params(elements)
