# Owner(s): ["module: dynamo"]
import torch
from torch._dynamo.device_interface import (
    CudaInterface,
    DeviceInterface,
    MtiaInterface,
    MpsInterface,
    XpuInterface,
    device_interfaces,
    register_interface_for_device,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestConstantFoldFnsRegistry(TestCase):
    def test_base_slots_are_empty(self):
        self.assertEqual(DeviceInterface.dynamo_constant_fold_fns, ())
        self.assertEqual(DeviceInterface.dynamo_constant_fold_fns_need_guards, ())

    def test_builtin_interfaces_declare_fold_fns(self):
        self.assertIn(torch.cuda.is_available, CudaInterface.dynamo_constant_fold_fns)
        self.assertIn(
            torch.cuda.current_device,
            CudaInterface.dynamo_constant_fold_fns_need_guards,
        )
        self.assertIn(torch.xpu.is_available, XpuInterface.dynamo_constant_fold_fns)
        self.assertIn(
            torch.xpu.current_device,
            XpuInterface.dynamo_constant_fold_fns_need_guards,
        )
        self.assertIn(torch.mps.is_available, MpsInterface.dynamo_constant_fold_fns)
        self.assertIn(torch.mtia.is_available, MtiaInterface.dynamo_constant_fold_fns)

    def test_tables_built_from_registry(self):
        from torch._dynamo.variables import torch as torch_variables

        self.assertIn(torch.cuda.is_available, torch_variables.constant_fold_functions)
        self.assertIn(
            torch.cuda.current_device, torch_variables.constant_fold_functions
        )
        self.assertIn(
            torch.cuda.current_device,
            torch_variables.constant_fold_functions_need_guards,
        )
        self.assertIn(torch.mps.is_available, torch_variables.constant_fold_functions)
        self.assertIn(
            torch._C._get_privateuse1_backend_name,
            torch_variables.constant_fold_functions,
        )

    def test_need_guards_entries_are_also_foldable(self):
        from torch._dynamo.variables import torch as torch_variables

        for fn in torch_variables.constant_fold_functions_need_guards:
            self.assertIn(fn, torch_variables.constant_fold_functions)

    def test_late_registration_rebuilds_tables(self):
        from torch._dynamo.variables.torch import (
            _rebuild_constant_fold_tables,
            constant_fold_functions,
        )

        def fake_fold_fn():
            return True

        class FakeFoldInterface(DeviceInterface):
            dynamo_constant_fold_fns = (fake_fold_fn,)

        register_interface_for_device("fake_fold_device", FakeFoldInterface)
        try:
            self.assertIn(fake_fold_fn, constant_fold_functions)
        finally:
            device_interfaces.pop("fake_fold_device", None)
            _rebuild_constant_fold_tables()
        self.assertNotIn(fake_fold_fn, constant_fold_functions)


if __name__ == "__main__":
    run_tests()
