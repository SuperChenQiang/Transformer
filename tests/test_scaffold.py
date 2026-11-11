"""工程与环境检查，不验证尚未实现的 Transformer 算法。

可使用 pytest；也可用标准库 unittest 执行此文件。
"""

import importlib
import pkgutil
import unittest
from pathlib import Path

import numpy as np
import torch

import transformer


class ScaffoldTests(unittest.TestCase):
    def test_imports_come_from_this_project(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "src" / "transformer" / "__init__.py"
        self.assertEqual(Path(transformer.__file__).resolve(), expected)

    def test_source_modules_are_importable(self) -> None:
        for module in pkgutil.walk_packages(transformer.__path__, prefix="transformer."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_torch_cpu_autograd(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor.square().sum().backward()
        torch.testing.assert_close(tensor.grad, torch.tensor([2.0, 4.0, 6.0]))

    def test_numpy_torch_interoperability(self) -> None:
        array = np.array([1.0, 2.0], dtype=np.float32)
        tensor = torch.from_numpy(array)
        np.testing.assert_array_equal(tensor.numpy(), array)


if __name__ == "__main__":
    unittest.main()
