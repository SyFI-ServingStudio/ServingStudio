"""Exercise service command wiring without Docker, tmux sessions or network calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
FAKE_COMMAND = r'''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
keys = ('VIBESIM_AGENT_MAIN_DIR', 'VIBESIM_AGENT_WORKSPACES_ROOT',
        'VIBESIM_AGENT_PROVIDERS_FILE', 'VIBESIM_AGENT_BIND', 'VIBESIM_AGENT_PORT',
        'VIBESIM_AGENT_ANALYZER_BASE_URL', 'VIBESIM_AGENT_MANAGED_BACKEND_URL',
        'VIBESIM_RUNNER_IMAGE', 'ANALYZER_PROXY_TARGET', 'CONVERSATION_PROXY_TARGET')
with open(os.environ['SERVICE_TEST_RECORD'], 'a') as stream:
    stream.write(json.dumps({'command': name, 'args': sys.argv[1:], 'cwd': os.getcwd(),
                            'env': {key: os.environ.get(key) for key in keys}}) + '\n')
if name == 'tmux':
    raise SystemExit(1)
if name == 'docker' and sys.argv[1:3] == ['network', 'inspect']:
    print('127.0.0.1')
if name == 'curl':
    print('200', end='')
'''


class ServiceCommandsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='vibesim-service-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'scripts').mkdir()
        for name in ('service-env.sh', 'run-service.sh', 'start-services.sh',
                     'services.sh', 'check-ports.py'):
            shutil.copyfile(SOURCE / 'scripts' / name, self.root / 'scripts' / name)
        shutil.copyfile(SOURCE / 'justfile', self.root / 'justfile')
        for name in ('scratch', 'cache', 'agent/vibesim_agent', 'sim', 'ui', 'bin'):
            (self.root / name).mkdir(parents=True)
        (self.root / 'agent/vibesim_agent/__main__.py').touch()
        (self.root / 'providers.yaml').write_text('version: 1\n')
        self.record = self.root / 'commands.jsonl'
        self.record.touch()
        settings = {
            'TMPDIR': self.root / 'scratch', 'UV_CACHE_DIR': self.root / 'cache',
            'VIBESIM_AGENT_DIR': self.root / 'agent', 'VIBESIM_SIM_DIR': self.root / 'sim',
            'VIBESIM_UI_DIR': self.root / 'ui', 'VIBESIM_ANALYZER_BIN': self.root / 'bin/analyze',
            'VIBESIM_AGENT_MAIN_DIR': self.root / 'sim',
            'VIBESIM_AGENT_WORKSPACES_ROOT': self.root / 'state',
            'VIBESIM_AGENT_PROVIDERS_FILE': self.root / 'providers.yaml',
            'VIBESIM_RUNNER_IMAGE': 'fixture-image', 'VIBESIM_PORT_BASE': '64000',
            'VIBESIM_API_BIND': '127.0.0.1',
        }
        # Test paths contain no shell metacharacters; values are still quoted.
        (self.root / '.env').write_text(''.join(f'export {key}="{value}"\n' for key, value in settings.items()))
        for name in ('uv', 'npm', 'analyze', 'docker', 'tmux', 'python3', 'curl'):
            path = self.root / 'bin' / name
            path.write_text(f'#!{sys.executable}\n' + FAKE_COMMAND)
            path.chmod(0o700)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(('VIBESIM_', 'CODEX_', 'AGENT_', 'ANALYZER_', 'UI_', 'CONVERSATION_'))
        }
        self.environment.update(PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'],
                                SERVICE_TEST_RECORD=str(self.record))

    def run_script(self, script, *args, success=True):
        result = subprocess.run(['bash', str(self.root / 'scripts' / script), *args],
                                cwd=self.root, env=self.environment, capture_output=True, text=True, timeout=10)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def commands(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def test_explicit_init_uses_selected_agent_and_state(self):
        self.run_script('services.sh', 'init')
        [call] = self.commands()
        self.assertEqual(call['command'], 'uv')
        self.assertEqual(call['args'], ['run', '--frozen', 'python', '-m', 'vibesim_agent', 'init'])
        self.assertEqual(call['cwd'], str(self.root / 'agent'))
        self.assertEqual(call['env']['VIBESIM_AGENT_WORKSPACES_ROOT'], str(self.root / 'state'))
        self.assertEqual(call['env']['VIBESIM_AGENT_PROVIDERS_FILE'], str(self.root / 'providers.yaml'))

    def test_backend_uses_new_runtime_configuration(self):
        self.run_script('run-service.sh', 'backend')
        [call] = self.commands()
        self.assertEqual(call['args'], ['run', '--frozen', 'python', '-m', 'vibesim_agent', 'serve'])
        self.assertEqual(call['env']['VIBESIM_AGENT_BIND'], '127.0.0.1')
        self.assertEqual(call['env']['VIBESIM_AGENT_PORT'], '64001')
        self.assertEqual(call['env']['VIBESIM_AGENT_ANALYZER_BASE_URL'], 'http://host.docker.internal:64002')
        self.assertEqual(call['env']['VIBESIM_AGENT_MANAGED_BACKEND_URL'], 'http://host.docker.internal:64001')

    def test_analyzer_uses_same_registry_and_selected_binary(self):
        self.run_script('run-service.sh', 'analyzer')
        [call] = self.commands()
        self.assertEqual(call['command'], 'analyze')
        self.assertEqual(call['cwd'], str(self.root / 'sim'))
        self.assertEqual(call['args'], ['serve', '--bind', '127.0.0.1:64002',
                                       '--workspace-registry', str(self.root / 'state/registry.json')])

    def test_ui_uses_existing_dev_script_and_both_proxy_targets(self):
        self.run_script('run-service.sh', 'frontend')
        [call] = self.commands()
        self.assertEqual(call['args'], ['run', 'dev', '--', '--port', '64000', '--strictPort'])
        self.assertEqual(call['cwd'], str(self.root / 'ui'))
        self.assertEqual(call['env']['CONVERSATION_PROXY_TARGET'], 'http://127.0.0.1:64001')
        self.assertEqual(call['env']['ANALYZER_PROXY_TARGET'], 'http://127.0.0.1:64002')

    def test_start_refuses_uninitialized_state_without_creating_it(self):
        result = self.run_script('start-services.sh', success=False)
        self.assertIn('just agent-init', result.stderr)
        self.assertFalse((self.root / 'state').exists())
        self.assertFalse(any('new-session' in call['args'] for call in self.commands()))

    def test_relative_state_path_is_rejected(self):
        with (self.root / '.env').open('a') as stream:
            stream.write('export VIBESIM_AGENT_WORKSPACES_ROOT=relative-state\n')
        result = self.run_script('services.sh', 'init', success=False)
        self.assertIn('must be an absolute path', result.stderr)
        self.assertEqual(self.commands(), [])

    def test_smoke_uses_versioned_endpoints(self):
        result = subprocess.run(['just', 'smoke', 'http://fixture.invalid'], cwd=self.root,
                                env=self.environment, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call['args'][-1] for call in self.commands()], [
            'http://fixture.invalid' + path for path in ('/', '/api/agent/v1/workspaces',
            '/api/analyzer/v1/runs', '/api/analyzer/v1/predictions',
            '/api/analyzer/v1/kernel-profiles', '/api/analyzer/v1/kernel-measurements')])

    def test_stop_remains_available_when_provider_file_is_missing(self):
        (self.root / 'providers.yaml').unlink()
        self.run_script('services.sh', 'stop')
        self.assertEqual([call['args'][-1] for call in self.commands()],
                         ['frontend', 'analyzer', 'backend'])


if __name__ == '__main__':
    unittest.main()
