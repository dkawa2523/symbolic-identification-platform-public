"""Exact pinned PhySO unit-stage plus four-file memory repair; no search changes."""
import argparse
import hashlib
import json
from pathlib import Path
import patch_physo_unit_roundoff as unit

BEFORE = {'physym/dimensional_analysis.py': '96c46d8a5a911568b1132bd2e2c8687d01b7814fa90d40482310572cca2c58ef',
 'physym/prior.py': 'adf0a33f9bb6e9625120ac6700a5c4ef2cbd20df492dbedf152c6f3e4f7ef734',
 'physym/token.py': 'aa6b47f10e980be69781cc2a6e24127b8cf80883649bcddeba85c7a87cf8fb00',
 'task/fit.py': '80efa6de7d8a40fa90aeccb3c4ffcee278c29d16b860b9ea6d535d83b9241cd6'}
AFTER = {'physym/dimensional_analysis.py': '89328c0568837e9b8c2763d5dbc4e726f264b4f4d2f2c2f8f800fa088a47f607',
 'physym/prior.py': '024fb6df7341929dc27be33a3ba222d034c47a733716ade6edcea50aa047c885',
 'physym/token.py': '1879f6dc6b7ada0a94d50308e4a6f83125ae27af2132687e047d80c95930303e',
 'task/fit.py': '343425dc560a78527c337971bb3a10a465008e52ea4e0ab5c3acc00d3b5b792c'}
REPLACEMENTS = {'physym/dimensional_analysis.py': [('            # Parsing previous token afterwards\n'
                                     '            parser(i - 1)\n'
                                     '        parser(start)\n'
                                     '    #t1 = time.perf_counter()\n'
                                     '    #print("assign_units_bottom_up %f ms"%((t1-t0)*1e3))\n',
                                     '            # Parsing previous token afterwards\n'
                                     '            parser(i - 1)\n'
                                     '        try:\n'
                                     '            parser(start)\n'
                                     '        finally:\n'
                                     '            # Break the recursive closure cycle without '
                                     'changing parsing order.\n'
                                     '            del parser, assign_units\n'
                                     '    #t1 = time.perf_counter()\n'
                                     '    #print("assign_units_bottom_up %f ms"%((t1-t0)*1e3))\n')],
 'physym/prior.py': [('        self.lib       = library\n'
                      '        self.progs     = programs\n'
                      '        self.get_default_mask_prob = lambda : '
                      'np.ones((self.progs.batch_size, self.lib.n_choices), dtype = float)\n'
                      '        self.reset_mask_prob()\n'
                      '\n'
                      '    def reset_mask_prob (self):\n',
                      '        self.lib       = library\n'
                      '        self.progs     = programs\n'
                      '        self.reset_mask_prob()\n'
                      '\n'
                      '    def get_default_mask_prob(self):\n'
                      '        # Do not store a self-capturing callable on self: it retains progs '
                      'until cyclic GC.\n'
                      '        return np.ones((self.progs.batch_size, self.lib.n_choices), '
                      'dtype=float)\n'
                      '\n'
                      '    def reset_mask_prob (self):\n'),
                     ('        # Method of programs returning [effectors_role] of token at step '
                      '(and filling non-existing relatives with\n'
                      '        # self.lib.invalid_idx). Adding new axis when necessary for problem '
                      'symmetry.\n'
                      '        get_relatives_idx_dict = {\n'
                      "            'ancestor' : lambda step: self.progs.get_ancestors_idx_of_step "
                      '(step, no_ancestor_idx_filler = self.lib.invalid_idx),                 # '
                      'Returns (batch_size, max_n_relatives)\n'
                      "            'parent'   : lambda step: self.progs.get_parent_idx_of_step    "
                      '(step, no_parent_idx_filler   = self.lib.invalid_idx)[:, np.newaxis],  # '
                      'Returns (batch_size, 1)\n'
                      "            'sibling'  : lambda step: self.progs.get_sibling_idx_of_step   "
                      '(step, no_sibling_idx_filler  = self.lib.invalid_idx)[:, np.newaxis],  # '
                      'Returns (batch_size, 1)\n'
                      '                              }\n'
                      '        self.get_relatives_idx = '
                      'get_relatives_idx_dict[self.effectors_role]                            # '
                      'Returns (batch_size, max_n_relatives)\n'
                      '\n'
                      '        # -------- CONSTRAINTS MASK  --------\n',
                      '        # Method of programs returning [effectors_role] of token at step '
                      '(and filling non-existing relatives with\n'
                      '        # self.lib.invalid_idx). Adding new axis when necessary for problem '
                      'symmetry.\n'
                      '        # Freeze the same relationship selected by the former constructor '
                      'lambda.\n'
                      '        self._relatives_role = self.effectors_role\n'
                      '\n'
                      '\n'
                      '        # -------- CONSTRAINTS MASK  --------\n'),
                     ('                                            dtype = float)\n'
                      '        self.count_max_violations[(self.effectors, self.targets)] = '
                      'self.max_nb_violations                 # (n_constraints,)\n'
                      '\n'
                      '    def __call__(self):\n',
                      '                                            dtype = float)\n'
                      '        self.count_max_violations[(self.effectors, self.targets)] = '
                      'self.max_nb_violations                 # (n_constraints,)\n'
                      '\n'
                      '    def get_relatives_idx(self, step):\n'
                      "        if self._relatives_role == 'ancestor':\n"
                      '            return self.progs.get_ancestors_idx_of_step(step, '
                      'no_ancestor_idx_filler=self.lib.invalid_idx)\n'
                      "        if self._relatives_role == 'parent':\n"
                      '            return self.progs.get_parent_idx_of_step(step, '
                      'no_parent_idx_filler=self.lib.invalid_idx)[:, np.newaxis]\n'
                      '        return self.progs.get_sibling_idx_of_step(step, '
                      'no_sibling_idx_filler=self.lib.invalid_idx)[:, np.newaxis]\n'
                      '\n'
                      '    def __call__(self):\n'),
                     ('        # Method of programs returning ancestors of token at step (and '
                      'filling non-existing relatives with\n'
                      '        # self.lib.invalid_idx).\n'
                      '        self.get_ancestors_idx = lambda step: '
                      'self.progs.get_ancestors_idx_of_step (\n'
                      '            step, no_ancestor_idx_filler = '
                      'self.lib.invalid_idx)               # Returns (batch_size, '
                      'max_n_ancestors)\n'
                      '\n'
                      '        # ---- PRIOR TEMPLATES ----\n',
                      '        # Method of programs returning ancestors of token at step (and '
                      'filling non-existing relatives with\n'
                      '        # self.lib.invalid_idx).\n'
                      '\n'
                      '        # ---- PRIOR TEMPLATES ----\n'),
                     ('        # Both template of priors in one array so it is ready to be sliced '
                      'with mask of True, False depending on prog\n'
                      '        self.template_prior = np.array([forbid_functions_prior, '
                      'allow_all_prior])                                        # (2, '
                      'lib.n_choices)\n'
                      '\n'
                      '    def __call__(self):\n',
                      '        # Both template of priors in one array so it is ready to be sliced '
                      'with mask of True, False depending on prog\n'
                      '        self.template_prior = np.array([forbid_functions_prior, '
                      'allow_all_prior])                                        # (2, '
                      'lib.n_choices)\n'
                      '\n'
                      '    def get_ancestors_idx(self, step):\n'
                      '        return self.progs.get_ancestors_idx_of_step(step, '
                      'no_ancestor_idx_filler=self.lib.invalid_idx)\n'
                      '\n'
                      '    def __call__(self):\n')],
 'physym/token.py': [('        self.siblings_pos       = np.full(shape=self.shape + '
                      '(MAX_NB_SIBLINGS,), fill_value=self.default_siblings_pos , dtype=int)\n'
                      '        self.children_pos       = np.full(shape=self.shape + '
                      '(MAX_NB_CHILDREN,), fill_value=self.default_children_pos , dtype=int)\n'
                      '        self.ancestors_pos      = np.full(shape=self.shape + '
                      '(self.shape[1], ),  fill_value=self.default_ancestors_pos, dtype=int)\n'
                      '\n'
                      '        # Token family relationships: numbers\n',
                      '        self.siblings_pos       = np.full(shape=self.shape + '
                      '(MAX_NB_SIBLINGS,), fill_value=self.default_siblings_pos , dtype=int)\n'
                      '        self.children_pos       = np.full(shape=self.shape + '
                      '(MAX_NB_CHILDREN,), fill_value=self.default_children_pos , dtype=int)\n'
                      '        # Ancestor entries contain only token positions or INVALID_POS. '
                      'Preserve\n'
                      '        # their integer values with narrower storage when the entire range '
                      'fits.\n'
                      '        ancestor_limits = np.iinfo(np.int32)\n'
                      '        ancestor_dtype = (np.int32 if ancestor_limits.min <= min(0, '
                      'self.default_ancestors_pos)\n'
                      '                          and max(self.shape[1] - 1, '
                      'self.default_ancestors_pos) <= ancestor_limits.max else int)\n'
                      '        self.ancestors_pos      = np.full(shape=self.shape + '
                      '(self.shape[1], ),  fill_value=self.default_ancestors_pos, '
                      'dtype=ancestor_dtype)\n'
                      '\n'
                      '        # Token family relationships: numbers\n')],
 'task/fit.py': [('\n'
                  '    cell      = cell_reseter ()\n'
                  '    optimizer = run_config["learning_config"]["get_optimizer"](cell)\n'
                  '\n',
                  '\n'
                  '    cell      = cell_reseter ()\n'
                  '    # The shape-probing batch is not used by the learner; do not retain its '
                  'token tables.\n'
                  '    del batch\n'
                  '    optimizer = run_config["learning_config"]["get_optimizer"](cell)\n'
                  '\n')]}
DIM = 'physym/dimensional_analysis.py'
RAW = dict(BEFORE, **{DIM: unit.UPSTREAM_SHA256})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unit_record(marker):
    record = json.loads(marker.read_text())
    if (record.get('original_sha256') != unit.UPSTREAM_SHA256
            or record.get('patched_sha256') != BEFORE[DIM]
            or record.get('absolute_unit_exponent_tolerance') != 1e-12
            or record.get('relative_tolerance') != 0):
        raise ValueError('Invalid historical unit-stage marker')
    return record


def apply(package_dir):
    root = Path(package_dir)
    marker = root / 'symbolic_runtime_patch.json'
    old_marker = root / 'physym/symbolic_unit_roundoff_patch.json'
    current = {n: digest((root / n).read_bytes()) for n in BEFORE}
    if current == AFTER:
        historical = unit_record(old_marker)
        expected = {'version': 1, 'unit_stage': historical,
                    'unit_marker_sha256': digest(old_marker.read_bytes()),
                    'before_sha256': BEFORE, 'after_sha256': AFTER}
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise ValueError('Final runtime files require matching chained marker')
        return expected
    if marker.exists() or current not in (BEFORE, RAW):
        raise ValueError('Unknown or mixed PhySO runtime source; refuse partial patch')
    if current == RAW:
        if old_marker.exists():
            raise ValueError('Raw source cannot carry an existing unit-stage marker')
        # Validate all four inputs before the historical helper writes anything.
        unit.apply(root)
    historical = unit_record(old_marker)
    prepared = {}
    for name, replacements in REPLACEMENTS.items():
        path = root / name
        if digest(path.read_bytes()) != BEFORE[name]:
            raise ValueError('Unexpected source after unit stage')
        source = path.read_text()
        for old, new in replacements:
            if source.count(old) != 1:
                raise ValueError(f'Pinned replacement is not unique: {name}')
            source = source.replace(old, new, 1)
        data = source.encode('utf-8')
        if digest(data) != AFTER[name]:
            raise ValueError(f'Unexpected prepared runtime hash: {name}')
        compile(source, str(path), 'exec')
        prepared[name] = data
    # All outputs validated before any memory-stage write. Interrupted mixtures
    # are rejected, never silently completed on a later invocation.
    for name, data in prepared.items():
        (root / name).write_bytes(data)
    record = {'version': 1, 'unit_stage': historical,
              'unit_marker_sha256': digest(old_marker.read_bytes()),
              'before_sha256': BEFORE, 'after_sha256': AFTER}
    marker.write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', required=True)
    print(json.dumps(apply(parser.parse_args().package_dir)))
