# Owner(s): ["module: dynamo"]
import torch
from torch._dynamo.device_interface import (
    CpuInterface,
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
            _rebuild_device_derived_tables,
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
            _rebuild_device_derived_tables()
        self.assertNotIn(fake_fold_fn, constant_fold_functions)


class TestSupportedCtxManagerClasses(TestCase):
    def test_base_slot_is_empty(self):
        self.assertEqual(DeviceInterface.dynamo_supported_ctx_manager_classes, ())

    def test_builtin_interfaces_declare_ctx_manager_classes(self):
        self.assertIn(
            torch.cuda.amp.autocast_mode.autocast,
            CudaInterface.dynamo_supported_ctx_manager_classes,
        )
        self.assertIn(
            torch.cuda.use_mem_pool,
            CudaInterface.dynamo_supported_ctx_manager_classes,
        )
        self.assertIn(
            torch.cpu.amp.autocast_mode.autocast,
            CpuInterface.dynamo_supported_ctx_manager_classes,
        )

    def test_table_built_from_registry(self):
        from torch._dynamo.variables import torch as torch_variables

        table = torch_variables.supported_ctx_manager_classes
        # Core entries are always present
        self.assertIn(torch.autograd.grad_mode.no_grad, table)
        self.assertIn(torch.amp.autocast_mode.autocast, table)
        # Device entries come from the registry
        self.assertIn(torch.cuda.amp.autocast_mode.autocast, table)
        self.assertIn(torch.cuda.use_mem_pool, table)
        self.assertIn(torch.cpu.amp.autocast_mode.autocast, table)

    def test_late_registration_rebuilds_table(self):
        from torch._dynamo.variables.torch import (
            _rebuild_device_derived_tables,
            supported_ctx_manager_classes,
        )

        def fake_ctx_manager():
            yield

        class FakeCtxInterface(DeviceInterface):
            dynamo_supported_ctx_manager_classes = (fake_ctx_manager,)

        register_interface_for_device("fake_ctx_device", FakeCtxInterface)
        try:
            self.assertIn(fake_ctx_manager, supported_ctx_manager_classes)
        finally:
            device_interfaces.pop("fake_ctx_device", None)
            _rebuild_device_derived_tables()
        self.assertNotIn(fake_ctx_manager, supported_ctx_manager_classes)


class TestSynchronizeFnsTable(TestCase):
    def test_base_slot_is_empty(self):
        self.assertEqual(DeviceInterface.dynamo_synchronize_fns, ())

    def test_builtin_interfaces_declare_synchronize_fns(self):
        self.assertEqual(
            CudaInterface.dynamo_synchronize_fns, (torch.cuda.synchronize,)
        )
        self.assertEqual(XpuInterface.dynamo_synchronize_fns, (torch.xpu.synchronize,))
        self.assertEqual(MpsInterface.dynamo_synchronize_fns, (torch.mps.synchronize,))
        self.assertEqual(CpuInterface.dynamo_synchronize_fns, (torch.cpu.synchronize,))

    def test_table_built_from_registry(self):
        from torch._dynamo.variables import torch as torch_variables

        table = torch_variables._synchronize_fn_to_device_type
        self.assertIsNone(table[torch.accelerator.synchronize])
        self.assertEqual(table[torch.cuda.synchronize], "cuda")
        self.assertEqual(table[torch.xpu.synchronize], "xpu")
        self.assertEqual(table[torch.mps.synchronize], "mps")
        self.assertEqual(table[torch.cpu.synchronize], "cpu")

    def test_late_registration_rebuilds_table(self):
        from torch._dynamo.variables.torch import (
            _rebuild_device_derived_tables,
            _synchronize_fn_to_device_type,
        )

        def fake_synchronize(device=None):
            pass

        class FakeSyncInterface(DeviceInterface):
            dynamo_synchronize_fns = (fake_synchronize,)

        register_interface_for_device("fake_sync_device", FakeSyncInterface)
        try:
            self.assertEqual(
                _synchronize_fn_to_device_type[fake_synchronize], "fake_sync_device"
            )
        finally:
            device_interfaces.pop("fake_sync_device", None)
            _rebuild_device_derived_tables()
        self.assertNotIn(fake_synchronize, _synchronize_fn_to_device_type)


class TestCurrentStreamFns(TestCase):
    def test_base_slot_is_empty(self):
        self.assertEqual(DeviceInterface.dynamo_current_stream_fns, ())

    def test_builtin_interfaces_declare_current_stream_fns(self):
        self.assertEqual(
            CudaInterface.dynamo_current_stream_fns, (torch.cuda.current_stream,)
        )
        self.assertEqual(
            XpuInterface.dynamo_current_stream_fns, (torch.xpu.current_stream,)
        )

    def test_table_built_from_registry(self):
        from torch._dynamo.variables import torch as torch_variables

        fns = torch_variables._current_stream_fns
        self.assertIn(torch.accelerator.current_stream, fns)
        self.assertIn(torch.cuda.current_stream, fns)
        self.assertIn(torch.xpu.current_stream, fns)

    def test_late_registration_rebuilds_table(self):
        from torch._dynamo.variables.torch import (
            _current_stream_fns,
            _rebuild_device_derived_tables,
        )

        def fake_current_stream(device=None):
            pass

        class FakeStreamInterface(DeviceInterface):
            dynamo_current_stream_fns = (fake_current_stream,)

        register_interface_for_device("fake_stream_device", FakeStreamInterface)
        try:
            self.assertIn(fake_current_stream, _current_stream_fns)
        finally:
            device_interfaces.pop("fake_stream_device", None)
            _rebuild_device_derived_tables()
        self.assertNotIn(fake_current_stream, _current_stream_fns)


if __name__ == "__main__":
    run_tests()
