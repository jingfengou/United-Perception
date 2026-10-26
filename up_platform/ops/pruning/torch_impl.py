from up_platform.dependencies import dependency


def set_iteration(scheduler, iteration):
    scheduler.iteration = iteration


def last_prune_step(scheduler):
    return scheduler.get_last_prune_step() if hasattr(scheduler, 'get_last_prune_step') else 0


def module_type(name):
    if name != 'FrozenBatchNorm2d':
        raise ValueError('Not a declared UP MSBench type: ' + name)
    return dependency('msbench.nn.modules').FrozenBatchNorm2d


def build_scheduler(config):
    # Validate the requested policy before constructing a scheduler or model.
    # A renamed fake-sparse class is not evidence of equivalent pruning rules.
    if config['scheduler'].get('type', 'BaseScheduler') != 'AmpereScheduler':
        prepare_module = dependency('msbench.prepare')
        for field, table_name in (('fake_sparse', 'FakeSparseTable'),
                                  ('mask_generator', 'MaskGeneratorTable')):
            table = getattr(prepare_module, table_name, None)
            requested = config[field]['type']
            if table is not None and requested not in table:
                raise RuntimeError(
                    f"pruning.build_scheduler: installed MSBench has no {field} {requested!r}. "
                    f"Available: {', '.join(sorted(table))}. Install the MSBench revision "
                    "matching this UP recipe; the platform will not substitute another policy.")
    return dependency('msbench.scheduler').build_sparse_scheduler(config)


def prepare(scheduler, model):
    return scheduler.prepare_sparse_model(model=model)


def prune(scheduler, model, iteration):
    return scheduler.prune_model(model, iteration)


def export(scheduler, model):
    return scheduler.export_sparse_model(model)


def sparse_rate(scheduler, model):
    return scheduler.compute_sparse_rate(model)


def set_lr_scheduler(scheduler, lr_scheduler):
    return scheduler.set_lr_scheduler(lr_scheduler)


def init_optimizer(scheduler, optimizer):
    return scheduler.init_optimizer(optimizer)

