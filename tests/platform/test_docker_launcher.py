"""Host launcher contract tests. Docker is replaced only in these argv tests."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'up_docker'


@unittest.skipIf(os.name == 'nt', 'The host launcher is a Linux Bash script')
class DockerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / 'work with spaces'
        self.data = self.root / 'data with spaces'
        self.work.mkdir()
        self.data.mkdir()
        binary = self.root / 'bin'
        binary.mkdir()
        fake = binary / 'docker'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$ARGV_FILE"\n'
                        'exit "${DOCKER_EXIT_CODE:-0}"\n')
        fake.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('UP_', 'MCDNN_')) and key not in
                    ('CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'DEFAULT_TASKS')}
        self.env.update(PATH=str(binary) + os.pathsep + os.environ['PATH'],
                        UP_WORK=str(self.work), UP_DATA=str(self.data),
                        ARGV_FILE=str(self.root / 'argv'))

    def run_launcher(self, *args, **environment):
        result = subprocess.run(['bash', str(SCRIPT), *args], cwd=self.work,
                                env=dict(self.env, **environment), text=True, capture_output=True)
        output = self.root / 'argv'
        argv = output.read_bytes().decode().rstrip('\0').split('\0') if output.exists() else []
        return result, argv

    def test_help_needs_no_docker_or_environment(self):
        result = subprocess.run(['/bin/bash', str(SCRIPT), '--help'], env={'PATH': '/usr/bin:/bin'},
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('train|eval|inference|check|python', result.stdout)
        self.assertFalse((self.root / 'argv').exists())

    def test_nvidia_mounts_and_arguments_are_not_shell_interpreted(self):
        config = 'config $(touch should-not-exist).yaml'
        tail = ['--config', config, '--launch', 'pytorch', '--ng', '1',
                '--opts', 'saver.save_dir', 'outputs/nvidia']
        result, argv = self.run_launcher('nvidia', 'train', *tail, UP_GPU='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(argv[argv.index('--gpus') + 1], 'device=1')
        self.assertIn(f'type=bind,src={self.work},dst=/work', argv)
        self.assertIn(f'type=bind,src={self.data},dst=/data,readonly', argv)
        index = argv.index('up-unified:local')
        self.assertEqual(argv[index + 1:], ['nvidia', 'train', *tail])
        self.assertFalse((self.work / 'should-not-exist').exists())
        self.assertNotIn('--device=/dev/mxcd', argv)

    def test_multi_gpu_csv_is_one_docker_argument(self):
        result, argv = self.run_launcher('nvidia', 'check', UP_GPU='0,1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(argv[argv.index('--gpus') + 1], '"device=0,1"')

    def test_metax_visibility_and_explicit_task_selection(self):
        result, argv = self.run_launcher('metax', 'check', '--no-device',
                                         UP_GPU='2,3', DEFAULT_TASKS='det_3d', UP_IMAGE='custom:tag')
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ('--device=/dev/dri', '--device=/dev/mxcd', 'CUDA_VISIBLE_DEVICES=2,3',
                      'DEFAULT_TASKS=det_3d', 'custom:tag'):
            self.assertIn(value, argv)
        self.assertNotIn('--gpus', argv)
        self.assertEqual(argv[-3:], ['metax', 'check', '--no-device'])

    def test_eval_keeps_training_route_and_opts_last(self):
        tail = ['--config', 'model.yaml', '--launch', 'pytorch', '--ng', '1',
                '--opts', 'saver.resume_model', 'weights.pth']
        result, argv = self.run_launcher('metax', 'eval', *tail)
        self.assertEqual(result.returncode, 0, result.stderr)
        index = argv.index('up-unified:local')
        self.assertEqual(argv[index + 1:], ['metax', 'train', '--evaluate', *tail])

    def test_inference_preserves_checkpoint_and_output(self):
        tail = ['--config', 'model.yaml', '--ckpt', 'trained weights.pth',
                '--img_path', 'images/', '--vis_dir', 'outputs/predictions']
        result, argv = self.run_launcher('nvidia', 'inference', *tail)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(argv[-len(tail) - 2:], ['nvidia', 'inference', *tail])

    def test_missing_dataset_does_not_create_or_start_anything(self):
        missing = self.root / 'missing'
        result, argv = self.run_launcher('nvidia', 'train', UP_DATA=str(missing))
        self.assertEqual(result.returncode, 2)
        self.assertIn('UP_DATA', result.stderr)
        self.assertFalse(missing.exists())
        self.assertEqual(argv, [])

    def test_dataset_mount_is_optional(self):
        result, argv = self.run_launcher('nvidia', 'python', '-c', 'print(1)', UP_DATA='')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('dst=/data' in value for value in argv))

    def test_invalid_selection_does_not_start_docker(self):
        for args, environment in ((['other', 'train'], {}), (['metax', 'delete'], {}),
                                  (['nvidia', 'train'], {'UP_GPU': 'all; touch unexpected'})):
            with self.subTest(args=args, environment=environment):
                result, argv = self.run_launcher(*args, **environment)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(argv, [])

    def test_numerical_controls_are_opt_in_and_vendor_scoped(self):
        result, argv = self.run_launcher('nvidia', 'check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('DETERMINISTIC' in value or 'CUBLAS' in value for value in argv))
        for vendor in ('nvidia', 'metax'):
            result, argv = self.run_launcher(vendor, 'check', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONHASHSEED='131', MCDNN_USE_DETERMINISTIC_ALGO='1', MCDNN_STABLE_RESULT_MOD='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('CUBLAS_WORKSPACE_CONFIG', argv)
            self.assertIn('PYTHONHASHSEED', argv)
            self.assertEqual('MCDNN_STABLE_RESULT_MOD' in argv, vendor == 'metax')

    def test_docker_failure_is_propagated(self):
        result, _ = self.run_launcher('nvidia', 'check', DOCKER_EXIT_CODE='37')
        self.assertEqual(result.returncode, 37)


if __name__ == '__main__':
    unittest.main()
