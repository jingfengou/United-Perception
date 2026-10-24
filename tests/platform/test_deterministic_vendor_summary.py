"""A partial trace or warn-only policy must not become a strict training claim."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from summarize_deterministic_vendors import check_policy, loss_comparison


class DeterministicVendorSummary(unittest.TestCase):
    def test_near_zero_losses_keep_original_absolute_tolerance(self):
        result=loss_comparison([1.,0.],[1.000001,0.00002])
        self.assertFalse(result['values_equal'])
        self.assertEqual(result['first_difference_step'],1)
        self.assertEqual(result['first_exceeding_tolerance_step'],2)
        self.assertEqual(result['exceeding_tolerance_steps'],1)

    def test_empty_or_short_trace_is_rejected(self):
        for reference,actual in (([],[]),([1.,2.],[1.])):
            with self.assertRaises(ValueError):
                loss_comparison(reference,actual)

    def test_warn_only_cannot_be_reported_as_strict(self):
        environment={'deterministic':True,'cudnn_deterministic':True,'deterministic_warn_only':False,
            'fill_uninitialized_memory':True,'cudnn_benchmark':False,'matmul_tf32':False,'cudnn_tf32':False,
            'controls':{'CUBLAS_WORKSPACE_CONFIG':':4096:8','MCDNN_STABLE_RESULT_MOD':None,
                'MCDNN_USE_DETERMINISTIC_ALGO':None,'PYTORCH_DISABLE_CUDA_CUDNN_TF32':None,
                'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':None}}
        check_policy(environment)
        actual=copy.deepcopy(environment)
        actual['deterministic_warn_only']=True
        with self.assertRaisesRegex(ValueError,'Strict'):
            check_policy(actual)
