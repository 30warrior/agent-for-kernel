"""Check declared boundary coverage against materialized inputs, before freezing."""


def boundary_lengths(unit, minimum=1, multiple=1, maximum=None, profile='compact'):
    anchors = {1, unit - 1, unit, unit + 1, unit + max(1, unit // 2), 4 * unit + 1}
    if profile == 'full':
        anchors.update({2 * unit - 1, 2 * unit, 2 * unit + 1, 4 * unit - 1, 4 * unit})
    elif profile != 'compact':
        raise ValueError('Unknown test coverage profile: ' + str(profile))
    # Keep source-constrained axes legal (e.g. channel multiples and convolution minima).
    values = {((max(minimum, n) + multiple - 1) // multiple) * multiple for n in anchors}
    return sorted(n for n in values if n >= minimum and (maximum is None or n <= maximum))


def coverage_requirements(metadata, profile='compact'):
    hardware = metadata.get('hardware', {})
    target = metadata['target']
    vector_bytes = 16 if target == 'neon' else hardware.get('cpu', {}).get('isa', {}).get('sve', {}).get('current_thread_vector_bytes')
    sizes = {'float32': 4, 'float64': 8, 'float16': 2, 'bfloat16': 2,
             'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1, 'bool': 1,
             'complex64': 8, 'complex128': 16}
    axes, unassessed = [], []
    for base, inputs in enumerate(metadata['cases']):
        for argument, tensor in enumerate(inputs):
            if not isinstance(tensor, dict) or 'shape' not in tensor:
                if isinstance(tensor, (list, tuple)) or isinstance(tensor, dict) and 'type' not in tensor:
                    unassessed.append(dict(base_case=base, input=argument, reason='Nested inputs are not covered by the axis checker'))
                continue
            item_bytes = sizes.get(tensor['dtype'].removeprefix('torch.'))
            unit = 32 if target == 'cuda' else max(1, vector_bytes // item_bytes) if vector_bytes and item_bytes else None
            units = [unit] if unit else [4, 8, 16]
            for axis in range(len(tensor['shape'])):
                axes.append(dict(base_case=base, input=argument, axis=axis, units=units,
                    unit_source='CUDA warp boundary guide (not an assumed kernel tile)' if target == 'cuda' else
                    'Observed vector bytes / dtype bytes; generated code must remain VL agnostic' if unit else
                    'Generic test sizes; actual SVE lane count unavailable',
                    lengths=sorted({n for v in units for n in boundary_lengths(v, profile=profile)}),
                    tail_lengths=sorted({n for v in units for n in boundary_lengths(v, profile=profile)
                                         if n % 4 != 0}),
                    layout_lengths=sorted({v + max(1, v // 2) for v in units}) if profile == 'compact' else None,
                    layouts=['contiguous', 'unit_stride', 'strided'] if len(tensor['shape']) > 1 else ['contiguous', 'unit_stride']))
    return dict(profile=profile, axes=axes, unassessed_inputs=unassessed,
        policy='Compact: cover boundary lengths across layouts jointly, plus the multi-vector remainder in each legal layout. Full: every length in every legal layout.',
        limitation='Coverage of declared boundaries is not proof of all shapes or candidate branches. Fixed axes, bounds and layout exemptions are agent claims about the reference contract, retained for review.')


def check_coverage(spec, cases, metadata, profile='compact'):
    requirements = coverage_requirements(metadata, profile)
    declarations = spec.get('coverage')
    if not isinstance(declarations, list):
        raise ValueError('A coverage list declaring every original top-level tensor axis is required')
    expected = {(r['base_case'], r['input'], r['axis']): r for r in requirements['axes']}
    declared, report = set(), []
    incomplete = False
    for plan in declarations:
        allowed = {'base_case', 'input', 'axis', 'kind', 'reason', 'minimum', 'maximum', 'multiple', 'layout_exemptions'}
        if not isinstance(plan, dict) or set(plan) - allowed:
            raise ValueError('Invalid coverage declaration fields')
        if any(type(plan.get(k)) is not int for k in ('base_case', 'input', 'axis')):
            raise ValueError('Coverage base_case/input/axis must be integers')
        key = tuple(plan[k] for k in ('base_case', 'input', 'axis'))
        if key not in expected or key in declared:
            raise ValueError('Unknown or duplicate coverage axis: ' + str(key))
        declared.add(key)
        if not isinstance(plan.get('reason'), str) or not plan['reason'].strip():
            raise ValueError('Every coverage axis requires a reference-source justification')
        if plan.get('kind') == 'fixed':
            if set(plan) - {'base_case', 'input', 'axis', 'kind', 'reason'}:
                raise ValueError('Fixed axes accept only selectors, kind and reason')
            original = metadata['cases'][key[0]][key[1]]['shape'][key[2]]
            if any(entry['base_case'] == key[0] and inputs[key[1]].shape[key[2]] != original
                   for entry, inputs in zip(spec['cases'], cases)):
                raise ValueError('An axis declared fixed changes size: ' + str(key))
            report.append(dict(declaration=plan, status='exempt_fixed_axis'))
            continue
        if plan.get('kind') != 'variable':
            raise ValueError('Coverage kind must be variable or fixed')
        minimum, multiple, maximum = plan.get('minimum', 1), plan.get('multiple', 1), plan.get('maximum')
        if type(minimum) is not int or minimum < 1 or type(multiple) is not int or multiple < 1 or (maximum is not None and (type(maximum) is not int or maximum < minimum)):
            raise ValueError('Invalid coverage dimension constraints')
        requirement = expected[key]
        lengths = sorted({n for unit in requirement['units'] for n in boundary_lengths(unit, minimum, multiple, maximum, profile)})
        if not lengths:
            raise ValueError('Variable axis has no legal boundary sizes: ' + str(key))
        exemptions = plan.get('layout_exemptions', {})
        if not isinstance(exemptions, dict) or set(exemptions) - set(requirement['layouts']) or any(not isinstance(v, str) or not v.strip() for v in exemptions.values()):
            raise ValueError('Layout exemptions must name supported layouts with source-based reasons')
        layouts = [layout for layout in requirement['layouts'] if layout not in exemptions]
        if not layouts:
            raise ValueError('Cannot exempt every layout of a variable axis')
        observed = {layout: set() for layout in layouts}
        for entry, inputs in zip(spec['cases'], cases):
            if entry['base_case'] != key[0]:
                continue
            tensor = inputs[key[1]]
            if tensor.dim() != len(metadata['cases'][key[0]][key[1]]['shape']):
                raise ValueError('Coverage dimensions must keep the declared input rank')
            descriptor = entry['inputs'][key[1]]
            # Zeros can hide dropped elements. Only nonzero-capable generated tensors count.
            distribution = descriptor.get('distribution')
            active = distribution in ('ones', 'uniform') or distribution == 'randint' and (descriptor.get('low', 0) != 0 or descriptor.get('high', 1) > 1) or distribution == 'normal' and (descriptor.get('std', 1) > 0 or descriptor.get('mean', 0) != 0)
            if not active:
                continue
            matches = dict(contiguous=tensor.is_contiguous(), unit_stride=tensor.stride(key[2]) == 1,
                           strided=tensor.stride(key[2]) > 1)
            for layout in layouts:
                if matches[layout]:
                    observed[layout].add(tensor.shape[key[2]])
        # A singleton axis has no stride-dependent traversal. Coupled shapes such
        # as square 1x1 matrices cannot produce a non-unit stride by transposing.
        layout_lengths = lengths
        if profile == 'compact':
            layout_lengths = sorted({((max(minimum, unit + max(1, unit // 2)) + multiple - 1) // multiple) * multiple
                                     for unit in requirement['units']})
            layout_lengths = [n for n in layout_lengths if maximum is None or n <= maximum]
            if not layout_lengths:
                layout_lengths = [lengths[-1]]
        required = {layout: [n for n in layout_lengths if layout != 'strided' or n > 1] for layout in layouts}
        if profile == 'compact':
            # Compact mode avoids the Cartesian product: each legal layout must
            # appear, while all boundary lengths are checked across their union.
            required = {layout: [] for layout in layouts}
        gaps = {layout: sorted(set(required[layout]) - observed[layout]) for layout in layouts
                if set(required[layout]) - observed[layout]}
        if profile == 'compact':
            gaps.update({layout: ['any_length'] for layout in layouts if not observed[layout]})
            covered = set().union(*observed.values())
            if set(lengths) - covered:
                gaps['any_layout'] = sorted(set(lengths) - covered)
        row = dict(declaration=plan, required_lengths=lengths,
                   required_lengths_by_layout=required,
                   observed_lengths={k: sorted(v) for k, v in observed.items()})
        report.append(row)
        if gaps:
            incomplete = True
    undeclared = sorted(set(expected) - declared)
    if undeclared:
        incomplete = True
    result = dict(status='passed' if not incomplete else 'partial', profile=profile, axes=report,
                  unassessed_inputs=requirements['unassessed_inputs'], limitation=requirements['limitation'])
    return result
